"""
AntEvolve Database Viewer
PyQt6 app to open .pkl database files and visualise the running max of program scores.
"""

import sys
import pickle
import time
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

# Add project src to path so antevolve models can be unpickled
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSplitter, QListWidget,
    QListWidgetItem, QStatusBar, QFrame, QSizePolicy, QGroupBox,
    QScrollArea, QTextEdit, QTabWidget, QStackedWidget, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QSize, QRectF, QPointF
from PyQt6.QtWidgets import QApplication as _QApp  # for clipboard
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPainter, QPen, QConicalGradient

import pyqtgraph as pg

# ─── Colour palette ───────────────────────────────────────────────────────────
BG_DARK      = "#0d1117"
BG_CARD      = "#161b22"
BG_PANEL     = "#1c2128"
ACCENT_BLUE  = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORG   = "#f78166"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"
BORDER       = "#30363d"

BTN_HEIGHT = 28

# ─── Helpers (mirrored from present_db.py) ───────────────────────────────────

def get_score(program: Any) -> float:
    """Extract a single float score from a Program (or dict) object."""
    # Pydantic model
    scores = getattr(program, "scores", None)
    if scores is None and isinstance(program, dict):
        scores = program.get("scores", {})
    if isinstance(scores, dict):
        if "score" in scores and scores["score"] is not None:
            return float(scores["score"])
        for v in scores.values():
            if isinstance(v, (int, float)) and v is not None:
                return float(v)
    for key in ("score", "auc", "AUC"):
        val = getattr(program, key, None) or (program.get(key) if isinstance(program, dict) else None)
        if val is not None:
            return float(val)
    return 0.0


def running_max(values: List[float]) -> List[float]:
    out, cur = [], float("-inf")
    for v in values:
        cur = max(cur, v)
        out.append(cur)
    return out


# Palette for per-key running max curves (cycles if more keys than colours)
_EXTRA_COLOURS = [
    "#e879f9",  # purple
    "#f59e0b",  # amber
    "#06b6d4",  # cyan
    "#f43f5e",  # rose
    "#a3e635",  # lime
    "#818cf8",  # indigo
    "#fb923c",  # orange
    "#34d399",  # emerald
]


def get_scores_by_key(programs: List[Any]) -> Dict[str, List[float]]:
    """
    Returns a dict mapping each score key found across all programs
    to the ordered list of that key's values (None → 0.0).
    Excludes the generic 'score' key to avoid duplicating the primary curve.
    """
    # Gather all keys (preserve insertion order)
    all_keys: List[str] = []
    seen: set = set()
    for p in programs:
        raw = getattr(p, 'scores', None)
        if raw is None and isinstance(p, dict):
            raw = p.get('scores', {})
        if isinstance(raw, dict):
            for k in raw:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

    result: Dict[str, List[float]] = {}
    for key in all_keys:
        series: List[float] = []
        for p in programs:
            raw = getattr(p, 'scores', None)
            if raw is None and isinstance(p, dict):
                raw = p.get('scores', {})
            val = raw.get(key) if isinstance(raw, dict) else None
            series.append(float(val) if val is not None else 0.0)
        result[key] = series
    return result


def load_pkl(path: str):
    """Load a pkl file. Returns the raw object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_programs(db_obj) -> List[Any]:
    """Pull programs out of whatever the pkl contains."""
    progs = None
    if hasattr(db_obj, "programs"):
        raw = db_obj.programs
        if isinstance(raw, dict):
            progs = list(raw.values())
        elif isinstance(raw, list):
            progs = raw
    elif isinstance(db_obj, dict):
        if "programs" in db_obj:
            inner = db_obj["programs"]
            progs = list(inner.values()) if isinstance(inner, dict) else inner
        else:
            progs = list(db_obj.values())
    if progs is None:
        progs = []
    # Sort by creation timestamp
    progs.sort(key=lambda p: (getattr(p, "created", None) or (p.get("created", 0) if isinstance(p, dict) else 0)))
    return progs


# ─── Background loader ───────────────────────────────────────────────────────

class LoaderThread(QThread):
    done    = pyqtSignal(object, str)   # (db_obj, filepath)
    error   = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            obj = load_pkl(self.path)
            self.done.emit(obj, self.path)
        except Exception as exc:
            self.error.emit(str(exc))


# ─── Spinner overlay ─────────────────────────────────────────────────────────

class SpinnerOverlay(QWidget):
    """Semi-transparent overlay with an animated arc spinner."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(16)          # ~60 fps
        self._timer.timeout.connect(self._tick)
        self.hide()

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._angle = 0
        self.show()
        self.raise_()
        self._resize_to_parent()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    # ── internals ─────────────────────────────────────────────────────────────

    def _tick(self):
        self._angle = (self._angle - 4) % 360
        self.update()

    def _resize_to_parent(self):
        if self.parent():
            self.setGeometry(self.parent().rect())

    def resizeEvent(self, event):
        self._resize_to_parent()
        super().resizeEvent(event)

    def showEvent(self, event):
        self._resize_to_parent()
        super().showEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ── dim background ────────────────────────────────────────────────────
        painter.fillRect(self.rect(), QColor(13, 17, 23, 210))   # BG_DARK 82%

        cx = self.width()  / 2
        cy = self.height() / 2
        r  = 44             # arc radius

        # ── outer track ring ─────────────────────────────────────────────────
        track_pen = QPen(QColor(48, 54, 61), 6)   # BORDER colour
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # ── spinning gradient arc ─────────────────────────────────────────────
        arc_rect = QRectF(cx - r, cy - r, r * 2, r * 2)

        grad = QConicalGradient(QPointF(cx, cy), self._angle)
        grad.setColorAt(0.0,  QColor(ACCENT_BLUE))      # bright head
        grad.setColorAt(0.75, QColor(ACCENT_BLUE[:-2] if ACCENT_BLUE.endswith("ff") else ACCENT_BLUE + "44"))  # fade tail
        grad.setColorAt(1.0,  QColor(ACCENT_BLUE + "00"))

        arc_pen = QPen(grad, 6)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        # Draw 270° arc starting from the current angle
        painter.drawArc(arc_rect,
                        int(self._angle * 16),
                        int(-270 * 16))

        # ── label ─────────────────────────────────────────────────────────────
        painter.setPen(QColor(TEXT_PRIMARY))
        font = QFont("Inter", 13, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(cx - 120, cy + r + 16, 240, 30),
            Qt.AlignmentFlag.AlignHCenter,
            "Loading database…"
        )
        painter.end()


# ─── Plot widget ─────────────────────────────────────────────────────────────

class EvolutionPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent, background=BG_CARD)
        self._setup_style()
        self._scatter = None
        self._rmax_curve = None

    def _setup_style(self):
        self.setLabel("left",  "Score",          color=TEXT_MUTED)
        self.setLabel("bottom", "Program Index",  color=TEXT_MUTED)
        self.setTitle("Running Maximum of Program Scores", color=TEXT_PRIMARY, size="14pt")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.getAxis("left").setTextPen(pg.mkPen(TEXT_MUTED))
        self.getAxis("bottom").setTextPen(pg.mkPen(TEXT_MUTED))
        self.getAxis("left").setPen(pg.mkPen(BORDER))
        self.getAxis("bottom").setPen(pg.mkPen(BORDER))

    def plot_data(
        self,
        scores: List[float],
        rmax: List[float],
        extra_rmax: Optional[Dict[str, List[float]]] = None,
    ):
        self.clear()
        xs = np.arange(len(scores))

        # Scatter – individual combined scores
        scatter = pg.ScatterPlotItem(
            x=xs, y=np.array(scores),
            size=6, pen=None,
            brush=pg.mkBrush(ACCENT_BLUE + "88"),
            name="All scores",
        )
        self.addItem(scatter)

        # Primary step curve – running max of combined score
        rmax_arr = np.array(rmax)
        primary = pg.PlotDataItem(
            x=xs, y=rmax_arr,
            pen=pg.mkPen(ACCENT_GREEN, width=2.5),
            stepMode="right",
        )
        self.addItem(primary)

        # Per-key running max curves (dashed, distinct colours)
        key_curves: List[tuple] = []   # (label, curve)
        if extra_rmax:
            for idx, (key, rmax_vals) in enumerate(extra_rmax.items()):
                colour = _EXTRA_COLOURS[idx % len(_EXTRA_COLOURS)]
                pen = pg.mkPen(colour, width=1.8,
                               style=Qt.PenStyle.DashLine)
                curve = pg.PlotDataItem(
                    x=np.arange(len(rmax_vals)),
                    y=np.array(running_max(rmax_vals)),
                    pen=pen,
                    stepMode="right",
                )
                self.addItem(curve)
                key_curves.append((key, curve))

        # Annotate global best on primary series
        if len(rmax):
            best_idx = int(np.argmax(scores))
            best_val = scores[best_idx]
            arrow = pg.ArrowItem(
                pos=(best_idx, best_val),
                angle=-90,
                headLen=14,
                brush=pg.mkBrush(ACCENT_ORG),
                pen=pg.mkPen(ACCENT_ORG),
            )
            self.addItem(arrow)
            label = pg.TextItem(
                f"Best: {best_val:.4f}",
                color=ACCENT_ORG,
                anchor=(0.5, 1.3),
            )
            label.setPos(best_idx, best_val)
            self.addItem(label)

        # Legend
        legend = self.addLegend(offset=(10, 10))
        legend.addItem(primary, "Running max (combined)")
        for lbl, c in key_curves:
            legend.addItem(c, f"Max [{lbl}]")
        legend.addItem(scatter, "All scores")


# ─── Stats panel ─────────────────────────────────────────────────────────────

class StatsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(270)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Database Stats")
        title.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(title)

        self._rows: Dict[str, QLabel] = {}
        for key in ("File", "Programs", "Best Score", "Avg Score",
                    "Std Dev", "Stage", "Islands", "DB Type"):
            row = self._make_row(key)
            layout.addWidget(row)

        layout.addStretch()

    def _make_row(self, key: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"background:{BG_PANEL}; border-radius:6px; padding:4px;")
        h = QHBoxLayout(frame)
        h.setContentsMargins(8, 6, 8, 6)
        lbl_key = QLabel(key)
        lbl_key.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
        lbl_val = QLabel("—")
        lbl_val.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:11px; font-weight:600;")
        lbl_val.setWordWrap(True)
        h.addWidget(lbl_key)
        h.addStretch()
        h.addWidget(lbl_val)
        self._rows[key] = lbl_val
        return frame

    def update_stats(self, db_obj, scores: List[float], filepath: str):
        fname = Path(filepath).name
        self._rows["File"].setText(fname[:28] + "…" if len(fname) > 28 else fname)
        self._rows["Programs"].setText(str(len(scores)))
        if scores:
            self._rows["Best Score"].setText(f"{max(scores):.5f}")
            self._rows["Avg Score"].setText(f"{float(np.mean(scores)):.5f}")
            self._rows["Std Dev"].setText(f"{float(np.std(scores)):.5f}")
        stage = getattr(db_obj, "stage", None)
        self._rows["Stage"].setText(str(stage).split(".")[-1] if stage else "—")
        islands = getattr(db_obj, "num_islands", None)
        self._rows["Islands"].setText(str(islands) if islands else "—")
        self._rows["DB Type"].setText(type(db_obj).__name__)

    def clear_stats(self):
        for lbl in self._rows.values():
            lbl.setText("—")


# ─── Program helpers ────────────────────────────────────────────────────────

def _get_program_string(program) -> str:
    content = getattr(program, 'content', None) or {}
    if isinstance(content, dict):
        return content.get('program', '')
    if isinstance(program, dict):
        inner = program.get('content', {})
        return inner.get('program', '') if isinstance(inner, dict) else ''
    return ''





# ─── Programs tab ─────────────────────────────────────────────────────────────

class ProgramsTab(QWidget):
    """Browse and inspect individual programs in the loaded database."""

    _LIMITS = [10, 20, 30, 0]   # 0 = All

    def __init__(self):
        super().__init__()
        self._programs: List[Any] = []
        self._current_content = ''
        self._limit: int = 0          # 0 = show all
        self._sort_by_score: bool = False  # False = sort by index
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left list ─────────────────────────────────────────────────────────
        left = QFrame()
        left.setFixedWidth(312)
        left.setStyleSheet(f'background:{BG_CARD}; border-right:1px solid {BORDER};')
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 10, 8, 8)
        lv.setSpacing(6)

        hdr = QLabel('Programs')
        hdr.setFont(QFont('Inter', 11, QFont.Weight.Bold))
        hdr.setStyleSheet(f'color:{TEXT_MUTED};')
        lv.addWidget(hdr)

        self._count_lbl = QLabel('No database loaded')
        self._count_lbl.setStyleSheet(f'color:{TEXT_MUTED}; font-size:10px;')
        lv.addWidget(self._count_lbl)

        best_btn = QPushButton('⭐  Select Best')
        best_btn.setFixedHeight(30)
        best_btn.setToolTip('Jump to the program with the highest score')
        best_btn.setStyleSheet(
            f'background:{BG_PANEL}; color:{ACCENT_GREEN}; border:1px solid {ACCENT_GREEN};'
            f'border-radius:5px; font-size:11px; font-weight:600;'
        )
        best_btn.clicked.connect(self._select_best)
        lv.addWidget(best_btn)

        # ── Top-N limit buttons ────────────────────────────────────────────────
        limit_hdr = QLabel('Show')
        limit_hdr.setStyleSheet(f'color:{TEXT_MUTED}; font-size:10px; font-weight:600;')
        lv.addWidget(limit_hdr)

        limit_row = QHBoxLayout()
        limit_row.setSpacing(4)
        self._limit_btns: Dict[int, QPushButton] = {}
        for n in self._LIMITS:
            label = f'Top {n}' if n > 0 else 'All'
            btn = QPushButton(label)
            btn.setFixedHeight(BTN_HEIGHT)
            btn.setCheckable(True)
            btn.setChecked(n == self._limit)
            btn.setStyleSheet(self._limit_btn_style(n == self._limit))
            btn.clicked.connect(lambda checked, lim=n: self._set_limit(lim))
            self._limit_btns[n] = btn
            limit_row.addWidget(btn)
        lv.addLayout(limit_row)

        # ── Sort buttons ───────────────────────────────────────────────────────
        sort_hdr = QLabel('Sort by')
        sort_hdr.setStyleSheet(f'color:{TEXT_MUTED}; font-size:10px; font-weight:600;')
        lv.addWidget(sort_hdr)

        sort_row = QHBoxLayout()
        sort_row.setSpacing(4)
        self._sort_idx_btn = QPushButton('Index')
        self._sort_idx_btn.setFixedHeight(BTN_HEIGHT)
        self._sort_idx_btn.setCheckable(True)
        self._sort_idx_btn.setChecked(True)
        self._sort_idx_btn.clicked.connect(lambda: self._set_sort(by_score=False))
        sort_row.addWidget(self._sort_idx_btn)

        self._sort_score_btn = QPushButton('Score ↓')
        self._sort_score_btn.setFixedHeight(BTN_HEIGHT)
        self._sort_score_btn.setCheckable(True)
        self._sort_score_btn.setChecked(False)
        self._sort_score_btn.clicked.connect(lambda: self._set_sort(by_score=True))
        sort_row.addWidget(self._sort_score_btn)
        lv.addLayout(sort_row)

        self._list = QListWidget()
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.currentRowChanged.connect(self._on_select)
        lv.addWidget(self._list, stretch=1)
        root.addWidget(left)

        # ── Right detail ──────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.setSpacing(8)

        # Metadata bar
        meta_frame = QFrame()
        meta_frame.setStyleSheet(
            f'background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:6px;'
        )
        mfl = QHBoxLayout(meta_frame)
        mfl.setContentsMargins(14, 8, 14, 8)
        mfl.setSpacing(20)
        self._meta_labels: Dict[str, QLabel] = {}
        for key in ('ID', 'Island', 'Gen', 'Score', 'Created', 'Parents'):
            col = QWidget()
            cl = QVBoxLayout(col)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            kl = QLabel(key)
            kl.setStyleSheet(f'color:{TEXT_MUTED}; font-size:10px;')
            vl = QLabel('—')
            vl.setStyleSheet(f'color:{TEXT_PRIMARY}; font-size:12px; font-weight:600;')
            cl.addWidget(kl)
            cl.addWidget(vl)
            mfl.addWidget(col)
            self._meta_labels[key] = vl
        mfl.addStretch()

        self._copy_btn = QPushButton('⎘  Copy Code')
        self._copy_btn.setFixedHeight(30)
        self._copy_btn.setToolTip('Copy program source code to clipboard')
        self._copy_btn.setStyleSheet(
            f'background:{BG_PANEL}; color:{ACCENT_BLUE}; border:1px solid {ACCENT_BLUE};'
            f'border-radius:5px; font-size:11px; font-weight:600; padding:0 10px;'
        )
        self._copy_btn.clicked.connect(self._copy_code)
        self._copy_btn.setEnabled(False)
        mfl.addWidget(self._copy_btn)
        rv.addWidget(meta_frame)

        # ── Stacked pages ─────────────────────────────────────────────────────
        self._stack = QStackedWidget()

        # Page 0 – small program: plain code view
        p0 = QWidget()
        p0l = QVBoxLayout(p0)
        p0l.setContentsMargins(0, 0, 0, 0)
        p0l.setSpacing(4)
        self._size_lbl = QLabel()
        self._size_lbl.setStyleSheet(f'color:{TEXT_MUTED}; font-size:10px;')
        p0l.addWidget(self._size_lbl)
        self._code_view = QTextEdit()
        self._code_view.setReadOnly(True)
        self._code_view.setFont(QFont('Monospace', 10))
        p0l.addWidget(self._code_view, stretch=1)
        self._stack.addWidget(p0)  # idx 0



        # Page 2 – large text: truncated + reveal button
        p2 = QWidget()
        p2l = QVBoxLayout(p2)
        p2l.setContentsMargins(0, 0, 0, 0)
        p2l.setSpacing(6)
        banner = QFrame()
        banner.setStyleSheet(
            f'background:{BG_PANEL}; border:1px solid {BORDER}; border-radius:6px;'
        )
        bl = QHBoxLayout(banner)
        bl.setContentsMargins(12, 6, 12, 6)
        self._large_desc = QLabel()
        self._large_desc.setStyleSheet(f'color:{TEXT_MUTED}; font-size:11px;')
        bl.addWidget(self._large_desc)
        bl.addStretch()
        show_btn = QPushButton('Show full program')
        show_btn.setFixedHeight(30)
        show_btn.clicked.connect(self._reveal_full)
        bl.addWidget(show_btn)
        p2l.addWidget(banner)
        self._large_view = QTextEdit()
        self._large_view.setReadOnly(True)
        self._large_view.setFont(QFont('Monospace', 10))
        p2l.addWidget(self._large_view, stretch=1)
        self._stack.addWidget(p2)  # idx 2

        rv.addWidget(self._stack, stretch=1)
        root.addWidget(right, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    # ── Button style helpers ──────────────────────────────────────────────────

    @staticmethod
    def _limit_btn_style(active: bool) -> str:
        if active:
            return (
                f'background:{ACCENT_BLUE}; color:#000; border:1px solid {ACCENT_BLUE};'
                f'border-radius:4px; font-size:10px; font-weight:700;'
            )
        return (
            f'background:{BG_PANEL}; color:{TEXT_MUTED}; border:1px solid {BORDER};'
            f'border-radius:4px; font-size:10px;'
        )

    @staticmethod
    def _sort_btn_style(active: bool) -> str:
        if active:
            return (
                f'background:{ACCENT_ORG}; color:#000; border:1px solid {ACCENT_ORG};'
                f'border-radius:4px; font-size:10px; font-weight:700;'
            )
        return (
            f'background:{BG_PANEL}; color:{TEXT_MUTED}; border:1px solid {BORDER};'
            f'border-radius:4px; font-size:10px;'
        )

    def _set_limit(self, limit: int):
        self._limit = limit
        for n, btn in self._limit_btns.items():
            btn.setChecked(n == limit)
            btn.setStyleSheet(self._limit_btn_style(n == limit))
        self._refresh_list()

    def _set_sort(self, by_score: bool):
        self._sort_by_score = by_score
        self._sort_idx_btn.setChecked(not by_score)
        self._sort_idx_btn.setStyleSheet(self._sort_btn_style(not by_score))
        self._sort_score_btn.setChecked(by_score)
        self._sort_score_btn.setStyleSheet(self._sort_btn_style(by_score))
        self._refresh_list()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_programs(self, programs: List[Any]):
        self._programs = programs
        self._refresh_list()

    def _refresh_list(self):
        """Re-render the list applying current sort and limit settings."""
        self._list.blockSignals(True)
        self._list.clear()

        # Build indexed list so we can preserve original index for display
        indexed = list(enumerate(self._programs))  # (original_idx, program)

        # Sort
        if self._sort_by_score:
            indexed.sort(key=lambda t: get_score(t[1]), reverse=True)

        # Limit
        if self._limit > 0:
            indexed = indexed[:self._limit]

        for orig_i, p in indexed:
            score = get_score(p)
            isl   = getattr(p, 'island_id', '?')
            gen   = getattr(p, 'generation', '?')
            item  = QListWidgetItem(f'#{orig_i:04d}  {score:.4f}  isl={isl}  g{gen}')
            item.setData(Qt.ItemDataRole.UserRole, orig_i)
            self._list.addItem(item)

        total = len(self._programs)
        shown = len(indexed)
        if total == 0:
            self._count_lbl.setText('No database loaded')
        elif shown < total:
            self._count_lbl.setText(f'Showing {shown} of {total} programs')
        else:
            self._count_lbl.setText(f"{total} program{'s' if total != 1 else ''}")

        self._list.blockSignals(False)
        if shown > 0:
            self._list.setCurrentRow(0)
            self._on_select(0)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_select(self, row: int):
        if row < 0:
            return
        item = self._list.item(row)
        if item is None:
            return
        orig_idx = item.data(Qt.ItemDataRole.UserRole)
        if 0 <= orig_idx < len(self._programs):
            self._show_program(self._programs[orig_idx])

    def _select_best(self):
        """Select the program with the highest combined score."""
        if not self._programs:
            return
        best_orig = max(range(len(self._programs)), key=lambda i: get_score(self._programs[i]))
        # Find the row in the current (possibly filtered/sorted) list
        for row in range(self._list.count()):
            if self._list.item(row).data(Qt.ItemDataRole.UserRole) == best_orig:
                self._list.setCurrentRow(row)
                self._list.scrollToItem(
                    self._list.item(row),
                    QListWidget.ScrollHint.PositionAtCenter,
                )
                return

    def _show_program(self, program):
        content_str = _get_program_string(program)
        size_bytes  = len(content_str.encode('utf-8'))
        size_kb     = size_bytes / 1024
        self._current_content = content_str

        # Metadata bar
        pid     = getattr(program, 'program_id', '?') or '?'
        isl     = getattr(program, 'island_id',  '?')
        gen     = getattr(program, 'generation', '?')
        score   = get_score(program)
        cr      = getattr(program, 'created', None)
        ts      = time.strftime('%Y-%m-%d %H:%M', time.localtime(cr)) if cr else '?'
        parents = getattr(program, 'parent_ids', []) or []
        self._meta_labels['ID'].setText(str(pid)[:14])
        self._meta_labels['Island'].setText(str(isl))
        self._meta_labels['Gen'].setText(str(gen))
        self._meta_labels['Score'].setText(f'{score:.5f}')
        self._meta_labels['Created'].setText(ts)
        self._meta_labels['Parents'].setText(str(len(parents)))

        # Enable copy button whenever there is text content
        self._copy_btn.setEnabled(bool(content_str))

        if size_bytes < 20 * 1024:
            self._size_lbl.setText(f'Size: {size_kb:.1f} KB')
            self._code_view.setPlainText(content_str)
            self._stack.setCurrentIndex(0)
        else:
            self._large_desc.setText(
                f'Large program ({size_kb:.0f} KB) — showing first 8 KB.'
            )
            self._large_view.setPlainText(content_str[:8192] + '\n\n… truncated')
            self._stack.setCurrentIndex(1)

    # ── Copy code ─────────────────────────────────────────────────────────────

    def _copy_code(self):
        """Copy the current program source to the system clipboard."""
        if not self._current_content:
            return
        clipboard = _QApp.clipboard()
        clipboard.setText(self._current_content)
        # Visual feedback: briefly change button label
        self._copy_btn.setText('✓  Copied!')
        self._copy_btn.setStyleSheet(
            f'background:{ACCENT_GREEN}; color:#000; border:1px solid {ACCENT_GREEN};'
            f'border-radius:5px; font-size:11px; font-weight:600; padding:0 10px;'
        )
        QTimer.singleShot(1500, self._reset_copy_btn)

    def _reset_copy_btn(self):
        self._copy_btn.setText('⎘  Copy Code')
        self._copy_btn.setStyleSheet(
            f'background:{BG_PANEL}; color:{ACCENT_BLUE}; border:1px solid {ACCENT_BLUE};'
            f'border-radius:5px; font-size:11px; font-weight:600; padding:0 10px;'
        )



    def _reveal_full(self):
        self._large_view.setPlainText(self._current_content)
        self._large_desc.setText('Full program loaded.')


# ─── Main window ─────────────────────────────────────────────────────────────

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
QPushButton {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {ACCENT_BLUE};
    color: #000;
    border-color: {ACCENT_BLUE};
}}
QPushButton:pressed {{ opacity: 0.8; }}
QPushButton#openBtn {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1f6feb, stop:1 #388bfd);
    border: none;
    font-weight: bold;
    font-size: 14px;
}}
QListWidget {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_PRIMARY};
    font-size: 12px;
}}
QListWidget::item:selected {{
    background-color: {ACCENT_BLUE};
    color: #000;
    border-radius: 4px;
}}
QStatusBar {{ color: {TEXT_MUTED}; font-size: 11px; }}
QSplitter::handle {{ background: {BORDER}; }}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    font-size: 12px;
    color: {TEXT_MUTED};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QTextEdit {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    color: {TEXT_MUTED};
    font-family: monospace;
    font-size: 11px;
    border-radius: 6px;
}}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius:6px; }}
QTabBar::tab {{
    background: {BG_PANEL};
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    padding: 6px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{ background: {BG_CARD}; color: {TEXT_PRIMARY}; }}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AntEvolve DB Analysis")
        self.resize(1400, 820)
        self.setStyleSheet(STYLESHEET)

        self._loader: Optional[LoaderThread] = None
        self._db_obj = None
        self._history: List[str] = []  # recently opened files

        self._build_ui()
        self._status("Ready — open a .pkl database file to begin.")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top toolbar
        root.addWidget(self._build_toolbar())

        # Main splitter: sidebar | content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Left sidebar
        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        # Right content (tabs)
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Tab 1: Plot  (use a QWidget that can host the spinner overlay)
        plot_tab = QWidget()
        pt_layout = QHBoxLayout(plot_tab)
        pt_layout.setContentsMargins(8, 8, 8, 8)
        pt_layout.setSpacing(8)
        self._plot = EvolutionPlot()
        pt_layout.addWidget(self._plot, stretch=1)
        self._stats = StatsPanel()
        pt_layout.addWidget(self._stats)
        self._tabs.addTab(plot_tab, "📈  Running Max")

        # Spinner lives as a child of plot_tab so it overlays the whole tab
        self._spinner = SpinnerOverlay(plot_tab)

        # Tab 2: Programs
        self._programs_tab = ProgramsTab()
        self._tabs.addTab(self._programs_tab, "🔬  Programs")

        # Tab 3: Raw info
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._tabs.addTab(self._info_text, "🗒  DB Info")

        splitter.addWidget(self._tabs)
        splitter.setSizes([220, 1180])

        root.addWidget(splitter, stretch=1)

        # Status bar
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

    def _build_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(60)
        bar.setStyleSheet(f"background:{BG_CARD}; border-bottom:1px solid {BORDER};")
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 8, 16, 8)

        logo = QLabel("⚡ AntEvolve DB Analysis")
        logo.setFont(QFont("Inter", 15, QFont.Weight.Bold))
        logo.setStyleSheet(f"color:{TEXT_PRIMARY};")
        h.addWidget(logo)
        h.addStretch()

        self._open_btn = QPushButton("Open .pkl File")
        self._open_btn.setObjectName("openBtn")
        self._open_btn.setFixedHeight(38)
        self._open_btn.clicked.connect(self._open_file_dialog)
        h.addWidget(self._open_btn)

        return bar

    def _build_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setFixedWidth(220)
        frame.setStyleSheet(f"background:{BG_CARD}; border-right:1px solid {BORDER};")
        v = QVBoxLayout(frame)
        v.setContentsMargins(10, 14, 10, 10)
        v.setSpacing(8)

        lbl = QLabel("Recent Files")
        lbl.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        v.addWidget(lbl)

        self._recent_list = QListWidget()
        self._recent_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._recent_list.itemDoubleClicked.connect(self._open_recent)
        v.addWidget(self._recent_list, stretch=1)

        return frame

    # ── File handling ─────────────────────────────────────────────────────────

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open AntEvolve Database",
            str(Path(__file__).resolve().parent.parent),
            "Pickle Files (*.pkl);;All Files (*)",
        )
        if path:
            self._load_file(path)

    def _open_recent(self, item: QListWidgetItem):
        self._load_file(item.data(Qt.ItemDataRole.UserRole))

    def _load_file(self, path: str):
        if self._loader and self._loader.isRunning():
            return
        self._status(f"Loading {Path(path).name} …")
        self._open_btn.setEnabled(False)
        self._tabs.setCurrentIndex(0)        # switch to plot tab so spinner is visible
        self._spinner.start()
        self._loader = LoaderThread(path)
        self._loader.done.connect(self._on_loaded)
        self._loader.error.connect(self._on_error)
        self._loader.start()

    def _on_loaded(self, db_obj, filepath: str):
        self._spinner.stop()
        self._db_obj = db_obj
        self._open_btn.setEnabled(True)

        # Update recent list
        if filepath not in self._history:
            self._history.insert(0, filepath)
            self._history = self._history[:12]
            self._recent_list.clear()
            for fp in self._history:
                item = QListWidgetItem(Path(fp).name)
                item.setData(Qt.ItemDataRole.UserRole, fp)
                item.setToolTip(fp)
                self._recent_list.addItem(item)

        # Extract & plot
        programs   = extract_programs(db_obj)
        scores     = [get_score(p) for p in programs]
        rmax       = running_max(scores)
        extra_rmax = get_scores_by_key(programs)

        self._plot.plot_data(scores, rmax, extra_rmax=extra_rmax)
        self._stats.update_stats(db_obj, scores, filepath)
        self._programs_tab.load_programs(programs)
        self._populate_info(db_obj, programs, filepath)

        n = len(programs)
        best = max(scores) if scores else 0.0
        self._status(
            f"Loaded {Path(filepath).name} — {n} programs — "
            f"best score: {best:.5f}"
        )

    def _on_error(self, msg: str):
        self._spinner.stop()
        self._open_btn.setEnabled(True)
        self._status(f"Error: {msg}")
        self._info_text.setPlainText(f"Failed to load file:\n\n{msg}")

    # ── Info tab ──────────────────────────────────────────────────────────────

    def _populate_info(self, db_obj, programs, filepath: str):
        lines = [
            f"File     : {filepath}",
            f"Type     : {type(db_obj).__name__}",
            f"Programs : {len(programs)}",
        ]
        for attr in ("backup_prefix", "stage", "num_islands",
                     "divergence_cutoff", "island_crossover_cutoff",
                     "program_count", "dynamic_filter"):
            val = getattr(db_obj, attr, None)
            if val is not None:
                lines.append(f"{attr:<25}: {val}")

        if programs:
            lines.append("")
            lines.append("── First 5 programs ─────────────────────────────")
            for p in programs[:5]:
                pid  = getattr(p, "program_id", "?")
                isl  = getattr(p, "island_id",  "?")
                gen  = getattr(p, "generation", "?")
                sc   = get_score(p)
                cr   = getattr(p, "created", None)
                ts   = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cr)) if cr else "?"
                lines.append(f"  id={str(pid)[:8]}  island={isl}  gen={gen}  score={sc:.4f}  created={ts}")

        self._info_text.setPlainText("\n".join(lines))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self._statusbar.showMessage(msg)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    pg.setConfigOptions(antialias=True, foreground=TEXT_PRIMARY, background=BG_DARK)
    app = QApplication(sys.argv)
    app.setApplicationName("AntEvolve DB Analysis")

    # Apply a dark palette globally
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG_DARK))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base,            QColor(BG_CARD))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button,          QColor(BG_PANEL))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT_BLUE))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()

    # If a path was passed on the CLI, open it immediately
    if len(sys.argv) > 1:
        win._load_file(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
