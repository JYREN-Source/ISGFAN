import os, glob, logging, random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------- Basic Parameters -----------------
BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
SOURCE_DIR, TARGET_DIR = [os.path.join(BASE_DIR, d) for d in ("1HP", "3HP")]
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_dataset.pt")

CLASSES = [
    "ball_07", "ball_14", "ball_21",
    "inner_07", "inner_14", "inner_21",
    "normal",
    "outer_07", "outer_14", "outer_21"
]

SIGNAL_LENGTH, OVERLAP, MAX_SEGMENTS_PER_FILE = 2048, 0, 210
# -------------------------------------------

# ----------------- Logging ---------------------
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.ERROR)
# -------------------------------------------


# ================ Datasets =====================
class BaseDataset(Dataset):
    def __init__(self, data_dir, signal_length=SIGNAL_LENGTH, overlap=OVERLAP):
        self.data_dir, self.signal_length, self.overlap = data_dir, signal_length, overlap
        self.file_paths = []

    def _scan_files(self):
        return [f for f in glob.glob(os.path.join(self.data_dir, "*.mat"))
                if self._get_class_from_file(f) is not None]

    def _get_class_from_file(self, file_path):
        return next((c for c in CLASSES if c in os.path.basename(file_path)), None)

    def _process_segment(self, segment, file_path):
        seg = torch.FloatTensor(segment).unsqueeze(0)
        if "HP" in file_path:
            snr_db = -8
            power  = torch.mean(seg ** 2)
            std    = torch.sqrt(power / (10 ** (snr_db / 10)))
            seg   += torch.randn_like(seg) * std
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        return seg

class SourceDataset(BaseDataset):
    def __init__(self, data_dir, **kw):
        super().__init__(data_dir, **kw)
        self.file_paths  = self._scan_files()
        self.file_labels = [CLASSES.index(self._get_class_from_file(f)) for f in self.file_paths]
        self.segment_idx = []
        try:
            self.segment_idx = self._build_index()
        except Exception as e:
            logger.error(f"SourceDataset build_index failed: {e}")

    def _build_index(self):
        idx = []
        for f_idx, path in enumerate(self.file_paths):
            try:
                sig = next(v.flatten() for k, v in sio.loadmat(path).items() if '_DE_time' in k)
                step = int(self.signal_length * (1 - self.overlap))
                nseg = min((len(sig) - self.signal_length) // step + 1, MAX_SEGMENTS_PER_FILE)
                idx.extend([(f_idx, i * step) for i in range(nseg)])
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        return idx

    def __len__(self): return len(self.segment_idx)

    def __getitem__(self, idx):
        f_idx, start = self.segment_idx[idx]
        sig = next(v.flatten() for k, v in sio.loadmat(self.file_paths[f_idx]).items() if '_DE_time' in k)
        seg = self._process_segment(sig[start:start+self.signal_length], self.file_paths[f_idx])
        return seg, self.file_labels[f_idx]

class TargetDataset(BaseDataset):
    def __init__(self, data_dir, include_labels=False, **kw):
        super().__init__(data_dir, **kw)

        self.include_labels = include_labels
        self.file_paths     = self._scan_files()

        if include_labels:
            self.file_labels = [CLASSES.index(self._get_class_from_file(f))
                                for f in self.file_paths]

        self.segment_idx = []                       # Placeholder to ensure the field exists
        try:
            self.segment_idx = self._build_index()  # Actually build the index
        except Exception as e:
            logger.error(f"TargetDataset build_index failed: {e}")

    def _build_index(self):
        idx = []
        for f_idx, path in enumerate(self.file_paths):
            try:
                sig = next(v.flatten() for k, v in sio.loadmat(path).items() if '_DE_time' in k)
                step = int(self.signal_length * (1 - self.overlap))
                nseg = min((len(sig) - self.signal_length) // step + 1, MAX_SEGMENTS_PER_FILE)
                for i in range(nseg):
                    start = i * step
                    idx.append((f_idx, start, self.file_labels[f_idx]) if self.include_labels
                               else (f_idx, start))
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        return idx

    def __len__(self): return len(self.segment_idx)

    def __getitem__(self, idx):
        if self.include_labels:
            f_idx, start, lab = self.segment_idx[idx]
        else:
            f_idx, start      = self.segment_idx[idx]
        sig = next(v.flatten() for k, v in sio.loadmat(self.file_paths[f_idx]).items() if '_DE_time' in k)
        seg = self._process_segment(sig[start:start+self.signal_length], self.file_paths[f_idx])
        return (seg, lab) if self.include_labels else seg


# ---------- Materialized Cached Dataset ----------
class MaterializedDataset(Dataset):
    def __init__(self, data_list, include_labels=True):
        self.data_list, self.include_labels = data_list, include_labels
    def __len__(self):  return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx] if self.include_labels else self.data_list[idx][0]
# ===================================


# ============== DataManager =========
class DataManager:
    _instance, _init_flag = None, False
    def __new__(cls):
        if not cls._instance: cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._init_flag:
            self.train_loader = self.val_loader = None
            self.target_loader = self.test_loader = None
            DataManager._init_flag = True

    def initialize(self, batch_size=32, num_workers=8, force_reload=False):
        if self.train_loader and not force_reload:
            return (self.train_loader,self.target_loader)

        print("\nInitializing data loaders ...")
        # ---------- Source Domain ----------
        full_ds   = SourceDataset(SOURCE_DIR)

        # ---------- Source Domain ----------
        self.train_loader = DataLoader(
            full_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=True
        )

        # ---------- Target Domain / Test Set ----------
        # +++ NEW: Check if test dataset file exists +++
        if os.path.exists(TEST_DATA_PATH) and not force_reload:
            print(f"\nLoading existing test dataset from {TEST_DATA_PATH}")
            cache = torch.load(TEST_DATA_PATH)
        else:
            print("\nCreating new test dataset with noise...")
            if os.path.exists(TEST_DATA_PATH):
                print("Overwriting existing test dataset...")
            tmp_ds     = TargetDataset(TARGET_DIR, include_labels=True)
            tmp_loader = DataLoader(tmp_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers, pin_memory=True)
            cache = []
            for seg, lab in tmp_loader:
                for s, l in zip(seg, lab):
                    cache.append((s, l))
            torch.save(cache, TEST_DATA_PATH)
            print(f"Test dataset saved to {TEST_DATA_PATH}")

        # a) Test Set (with labels)
        test_ds = MaterializedDataset(cache, include_labels=True)
        self.test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )

        # b) Target Domain (without labels)
        tgt_ds = MaterializedDataset(cache, include_labels=False)
        self.target_loader = DataLoader(
            tgt_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=True
        )

        print("\nData Statistics")
        print(f"Source Train : {len(full_ds)}")
        print(f"Target (no y): {len(tgt_ds)}")
        print(f"Test (y)     : {len(test_ds)}")

        return (self.train_loader,self.target_loader)

data_manager = DataManager()

def get_dataloaders(batch_size=32, num_workers=8):
    return data_manager.initialize(batch_size, num_workers)
# ===================================


# ---------------- Val ----------------
if __name__ == "__main__":
    train_loader, target_loader = get_dataloaders()

    x, y = next(iter(train_loader))
    print(f"Train batch : {x.shape} , {y.shape}")

    x_tgt = next(iter(target_loader))
    print(f"Target batch: {x_tgt.shape}")

    _, test_loader = data_manager.test_loader, data_manager.test_loader
    if test_loader:
        x_test, y_test = next(iter(test_loader))
        print(f"Test batch  : {x_test.shape} , {y_test.shape}")