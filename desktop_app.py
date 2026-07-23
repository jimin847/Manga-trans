#!/usr/bin/env python3
"""
Manga-trans Desktop Application (PyQt5) — Direct Pipeline Execution
차분하고 직관적인 데스크탑 유틸리티 GUI. PyInstaller 서브프로세스 충돌 없이 QThread 내부에서 직접 번역 파이프라인 구동.
"""
import glob
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

from PyQt5.QtCore import QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_PATH = REPO_ROOT / ".env"

# ── 차분한 다크 유틸리티 QSS ─────────────────────────────────────────────
DARK_UTILITY_QSS = """
QMainWindow, QDialog {
    background-color: #111215;
}
QWidget {
    color: #ededed;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
}
QFrame#HeaderFrame {
    background-color: #111215;
    border-bottom: 1px solid #292b32;
}
QLabel#AppTitle {
    font-size: 14px;
    font-weight: 600;
    color: #ededed;
}
QLabel#AppSubtitle {
    font-size: 14px;
    font-weight: 400;
    color: #5d6169;
}
QLabel#StatusText {
    font-size: 12px;
    color: #8d929a;
}
QFrame#TabBarFrame {
    background-color: #17181c;
    border: 1px solid #292b32;
    border-radius: 6px;
}
QPushButton.TabBtn {
    background-color: transparent;
    border: none;
    color: #8d929a;
    padding: 6px 14px;
    border-radius: 4px;
    font-weight: 500;
}
QPushButton.TabBtn:hover {
    color: #ededed;
}
QPushButton.TabBtnActive {
    background-color: #1f2126;
    color: #ededed;
    border: none;
    padding: 6px 14px;
    border-radius: 4px;
    font-weight: 600;
}
QFrame#DropZone {
    background-color: #17181c;
    border: 1px dashed #292b32;
    border-radius: 8px;
}
QFrame#DropZone:hover {
    background-color: #1f2126;
    border: 1px dashed #8d929a;
}
QLabel#DropTitle {
    font-size: 14px;
    font-weight: 500;
    color: #ededed;
}
QLabel#DropSub {
    font-size: 12px;
    color: #8d929a;
}
QFrame#CardPanel {
    background-color: #17181c;
    border: 1px solid #292b32;
    border-radius: 8px;
}
QListWidget {
    background-color: #17181c;
    border: 1px solid #292b32;
    border-radius: 8px;
    outline: none;
}
QListWidget::item {
    border-bottom: 1px solid #292b32;
    padding: 12px 14px;
}
QListWidget::item:selected {
    background-color: #1f2126;
}
QCheckBox {
    font-size: 12px;
    color: #8d929a;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #5d6169;
    border-radius: 4px;
    background-color: #17181c;
}
QCheckBox::indicator:checked {
    background-color: #4e82ee;
    border-color: #4e82ee;
}
QLineEdit {
    background-color: #17181c;
    border: 1px solid #292b32;
    border-radius: 6px;
    padding: 8px 10px;
    color: #ededed;
}
QLineEdit:focus {
    border-color: #4e82ee;
}
QPushButton#PrimaryBtn {
    background-color: #ffffff;
    color: #000000;
    border: none;
    border-radius: 6px;
    padding: 10px 18px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton#PrimaryBtn:hover {
    background-color: #e6e6e6;
}
QPushButton#SecondaryBtn {
    background-color: #1f2126;
    color: #ededed;
    border: 1px solid #292b32;
    border-radius: 6px;
    padding: 10px 18px;
    font-weight: 500;
    font-size: 13px;
}
QPushButton#SecondaryBtn:hover {
    background-color: #26282f;
}
QProgressBar {
    border: none;
    background-color: #1f2126;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #4e82ee;
    border-radius: 4px;
}
"""


def load_env_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if key:
        return key
    if ENV_PATH.exists():
        try:
            for line in ENV_PATH.read_text("utf-8").splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


def save_env_api_key(key: str):
    os.environ["OPENROUTER_API_KEY"] = key
    lines = []
    if ENV_PATH.exists():
        try:
            lines = [l for l in ENV_PATH.read_text("utf-8").splitlines() if not l.strip().startswith("OPENROUTER_API_KEY=")]
        except Exception:
            pass
    lines.append(f'OPENROUTER_API_KEY="{key}"')
    try:
        ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"Error writing .env: {e}")


# ── 드래그 앤 드롭 전용 프레임 ──────────────────────────────────────────
class DropZoneFrame(QFrame):
    folders_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 36, 24, 36)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("번역할 만화 폴더를 여기에 끌어다 놓으세요 (Drag & Drop)")
        title.setObjectName("DropTitle")
        title.setAlignment(Qt.AlignCenter)

        sub = QLabel("또는 클릭하여 Finder에서 폴더를 직접 선택하세요")
        sub.setObjectName("DropSub")
        sub.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)
        layout.addSpacing(6)
        layout.addWidget(sub)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("background-color: #1f2126; border: 1px dashed #4e82ee;")

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event):
        self.setStyleSheet("")
        paths = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if os.path.isdir(local_path):
                paths.append(local_path)
        if paths:
            self.folders_dropped.emit(paths)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            folder = QFileDialog.getExistingDirectory(self, "만화 원서 폴더 선택")
            if folder:
                self.folders_dropped.emit([folder])


# ── QThread 내부 직접 파이프라인 구동 스레드 (서브프로세스 윈도우 중복 열림 방지) ──
class RealBatchWorkerThread(QThread):
    progress_updated = pyqtSignal(str, int, int, float, str)  # folder, current, total, speed, eta
    log_emitted = pyqtSignal(str)
    all_completed = pyqtSignal(int, int, int)  # success, cache, error

    def __init__(self, folders: list, api_key: str, skip_existing: bool):
        super().__init__()
        self.folders = folders
        self.api_key = api_key
        self.skip_existing = skip_existing
        self.is_running = True

    def run(self):
        total_success = 0
        total_cache = 0
        total_error = 0

        out_base_dir = REPO_ROOT / "output"
        os.makedirs(out_base_dir, exist_ok=True)

        try:
            from main import load_config, process_page
            from run_batch import load_progress, save_progress
        except Exception as e:
            self.log_emitted.emit(f"❌ 파이프라인 모듈 로드 실패: {e}")
            self.all_completed.emit(0, 0, 1)
            return

        config_path = REPO_ROOT / "config.yaml"
        config = load_config(str(config_path))

        for folder in self.folders:
            if not self.is_running:
                break

            folder_name = os.path.basename(folder) or folder
            images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                images.extend(glob.glob(os.path.join(folder, ext)))
            images.sort()

            total_pages = len(images)
            if total_pages == 0:
                self.log_emitted.emit(f"⚠️ 폴더 '{folder_name}' 내에 이미지 파일이 없습니다.")
                continue

            folder_out_dir = out_base_dir / folder_name
            os.makedirs(folder_out_dir, exist_ok=True)
            config["output"]["base_dir"] = str(folder_out_dir)

            resume_file = str(folder_out_dir / ".batch_progress.json")
            progress = load_progress(resume_file) if self.skip_existing else {}

            filtered_images = []
            for img in images:
                page_id = Path(img).stem
                if self.skip_existing and page_id in progress.get("completed", []):
                    total_cache += 1
                    continue
                if self.skip_existing and (folder_out_dir / f"{page_id}_ko.png").exists():
                    total_cache += 1
                    continue
                filtered_images.append(img)

            self.log_emitted.emit(f"▶ [시작] 폴더 '{folder_name}' (총 {total_pages}장 중 번역 대상 {len(filtered_images)}장)...")

            batch_start = time.time()
            results = list(progress.get("results", []))
            global_context = list(progress.get("context", []))

            for idx, img_path in enumerate(filtered_images, 1):
                if not self.is_running:
                    self.log_emitted.emit("⏹ 사용자에 의해 작업이 중단되었습니다.")
                    break

                page_id = Path(img_path).stem
                t0 = time.time()

                elapsed = time.time() - batch_start
                avg_per_img = elapsed / idx if idx > 0 else 0
                rem_sec = (len(filtered_images) - idx) * avg_per_img if avg_per_img else 0
                speed = (idx / elapsed) * 60 if elapsed > 0 else 0

                rm, rs = divmod(int(rem_sec), 60)
                rh, rm = divmod(rm, 60)
                if rh > 0:
                    eta_str = f"약 {rh}시간 {rm}분"
                elif rm > 0:
                    eta_str = f"약 {rm}분 {rs}초"
                else:
                    eta_str = f"약 {rs}초"

                self.progress_updated.emit(folder_name, idx, len(filtered_images), round(speed, 1), eta_str)
                self.log_emitted.emit(f"── [{idx}/{len(filtered_images)}] 페이지 '{page_id}' 분석 및 번역 중...")

                try:
                    res = process_page(img_path, config, self.api_key, previous_context=list(global_context))
                    page_time = time.time() - t0
                    res["_batch_time_s"] = round(page_time, 1)
                    status = res.get("status", "complete")
                    icon = "✅" if status == "complete" else "⚠️"

                    self.log_emitted.emit(f"  {icon} 처리 성공 ({page_time:.1f}초 경과)")
                    results.append(res)
                    total_success += 1

                    if "translations" in res:
                        global_context.extend(res["translations"])
                        if len(global_context) > 10:
                            global_context = global_context[-10:]

                    progress["completed"] = list(set(progress.get("completed", []) + [page_id]))
                    progress["results"] = results
                    progress["context"] = list(global_context)
                    save_progress(progress, resume_file)

                except Exception as e:
                    page_time = time.time() - t0
                    self.log_emitted.emit(f"  ❌ 실패 ({page_time:.1f}초): {e}")
                    total_error += 1
                    res = {
                        "page_id": page_id,
                        "status": "error",
                        "error": str(e),
                        "_batch_time_s": round(page_time, 1),
                    }
                    results.append(res)
                    progress["failed"] = list(set(progress.get("failed", []) + [page_id]))
                    progress["results"] = results
                    save_progress(progress, resume_file)

        self.all_completed.emit(total_success, total_cache, total_error)

    def stop(self):
        self.is_running = False


# ── API Key 설정 다이얼로그 ──────────────────────────────────────────────
class ApiKeyDialog(QDialog):
    def __init__(self, current_key="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenRouter API Key 설정")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        info = QLabel("OpenRouter API Key를 입력하세요.\n입력한 키는 로컬 .env 파일에 안전하게 저장됩니다.")
        info.setStyleSheet("color: #8d929a; font-size: 12px;")
        layout.addWidget(info)

        self.txt_key = QLineEdit(current_key)
        self.txt_key.setEchoMode(QLineEdit.Password)
        self.txt_key.setPlaceholderText("sk-or-v1-...")
        layout.addWidget(self.txt_key)

        btn_box = QHBoxLayout()
        btn_cancel = QPushButton("취소")
        btn_cancel.setObjectName("SecondaryBtn")
        btn_cancel.clicked.connect(self.reject)

        btn_save = QPushButton("저장")
        btn_save.setObjectName("PrimaryBtn")
        btn_save.clicked.connect(self.accept)

        btn_box.addStretch()
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_save)
        layout.addLayout(btn_box)

    def get_key(self):
        return self.txt_key.text().strip()


# ── 메인 윈도우 애플리케이션 ─────────────────────────────────────────────
class MangaTransDesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga-trans Desktop Pipeline")
        self.resize(800, 640)
        self.setMinimumSize(700, 540)

        self.registered_folders = []
        self.worker = None
        self.api_key = load_env_api_key()

        self.init_ui()
        self.check_system_health()

    def init_ui(self):
        central_w = QWidget()
        self.setCentralWidget(central_w)
        main_layout = QVBoxLayout(central_w)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 상단 헤더 바
        header_frame = QFrame()
        header_frame.setObjectName("HeaderFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(24, 14, 24, 14)

        title_lbl = QLabel("Manga-trans")
        title_lbl.setObjectName("AppTitle")
        sub_lbl = QLabel("Pipeline")
        sub_lbl.setObjectName("AppSubtitle")
        title_box = QHBoxLayout()
        title_box.setSpacing(6)
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)

        self.lama_status = QLabel("● Inpainting: LaMa ONNX [확인 중]")
        self.lama_status.setObjectName("StatusText")

        self.api_status = QLabel("● API Key [확인 중]")
        self.api_status.setObjectName("StatusText")

        btn_api_key = QPushButton("🔑 API Key 설정")
        btn_api_key.setObjectName("SecondaryBtn")
        btn_api_key.setCursor(Qt.PointingHandCursor)
        btn_api_key.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_api_key.clicked.connect(self.open_api_key_dialog)

        status_box = QHBoxLayout()
        status_box.setSpacing(14)
        status_box.addWidget(self.lama_status)
        status_box.addWidget(self.api_status)
        status_box.addWidget(btn_api_key)

        header_layout.addLayout(title_box)
        header_layout.addStretch()
        header_layout.addLayout(status_box)
        main_layout.addWidget(header_frame)

        # 2. 메인 컨텐츠 영역
        content_w = QWidget()
        content_layout = QVBoxLayout(content_w)
        content_layout.setContentsMargins(28, 20, 28, 20)
        content_layout.setSpacing(16)

        # 상단 네비게이션 탭
        tab_frame = QFrame()
        tab_frame.setObjectName("TabBarFrame")
        tab_layout = QHBoxLayout(tab_frame)
        tab_layout.setContentsMargins(4, 4, 4, 4)
        tab_layout.setSpacing(4)

        self.btn_tab1 = QPushButton("1. 작업 등록")
        self.btn_tab2 = QPushButton("2. 번역 진행 중")
        self.btn_tab3 = QPushButton("3. 작업 완료")

        self.btn_tab1.clicked.connect(lambda: self.switch_view(0))
        self.btn_tab2.clicked.connect(lambda: self.switch_view(1))
        self.btn_tab3.clicked.connect(lambda: self.switch_view(2))

        tab_layout.addWidget(self.btn_tab1)
        tab_layout.addWidget(self.btn_tab2)
        tab_layout.addWidget(self.btn_tab3)
        content_layout.addWidget(tab_frame, alignment=Qt.AlignHCenter)

        # 스택 위젯
        self.stack = QStackedWidget()
        self.setup_view1_input()
        self.setup_view2_running()
        self.setup_view3_completed()
        content_layout.addWidget(self.stack)

        main_layout.addWidget(content_w)
        self.switch_view(0)

    # ── View 1: 작업 등록 ────────────────────────────────────────────────
    def setup_view1_input(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self.drop_zone = DropZoneFrame()
        self.drop_zone.folders_dropped.connect(self.add_folders)
        layout.addWidget(self.drop_zone)

        list_header = QHBoxLayout()
        self.lbl_folder_count = QLabel("등록된 폴더 목록 (0개)")
        self.lbl_folder_count.setStyleSheet("color: #8d929a; font-weight: 500; font-size: 12px;")
        self.lbl_page_count = QLabel("총 0 페이지")
        self.lbl_page_count.setStyleSheet("color: #8d929a; font-weight: 500; font-size: 12px;")
        list_header.addWidget(self.lbl_folder_count)
        list_header.addStretch()
        list_header.addWidget(self.lbl_page_count)
        layout.addLayout(list_header)

        self.folder_list_widget = QListWidget()
        layout.addWidget(self.folder_list_widget)

        opts_layout = QHBoxLayout()
        opts_layout.setSpacing(20)
        self.chk_skip_existing = QCheckBox("기존 번역 완료된 파일 건너뛰기 (--skip-existing)")
        self.chk_skip_existing.setChecked(True)
        opts_layout.addWidget(self.chk_skip_existing)
        opts_layout.addStretch()
        layout.addLayout(opts_layout)

        btn_layout = QHBoxLayout()
        btn_clear = QPushButton("목록 비우기")
        btn_clear.setObjectName("SecondaryBtn")
        btn_clear.clicked.connect(self.clear_folders)

        btn_start = QPushButton("실제 번역 시작")
        btn_start.setObjectName("PrimaryBtn")
        btn_start.clicked.connect(self.start_batch)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_start)
        layout.addLayout(btn_layout)

        self.stack.addWidget(w)

    # ── View 2: 진행 중 ──────────────────────────────────────────────────
    def setup_view2_running(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        card = QFrame()
        card.setObjectName("CardPanel")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(10)

        top_row = QHBoxLayout()
        self.lbl_cur_folder = QLabel("진행 대기 중...")
        self.lbl_cur_folder.setStyleSheet("font-size: 15px; font-weight: 500; color: #ededed;")
        self.lbl_cur_ratio = QLabel("0 / 0 페이지 (0%)")
        self.lbl_cur_ratio.setStyleSheet("font-size: 13px; color: #8d929a;")
        top_row.addWidget(self.lbl_cur_folder)
        top_row.addStretch()
        top_row.addWidget(self.lbl_cur_ratio)
        card_layout.addLayout(top_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        card_layout.addWidget(self.progress_bar)

        detail_row = QHBoxLayout()
        self.lbl_speed = QLabel("처리 속도: 0.0 pages/min")
        self.lbl_speed.setStyleSheet("font-size: 12px; color: #5d6169;")
        self.lbl_eta = QLabel("예상 남은 시간: 계산 중...")
        self.lbl_eta.setStyleSheet("font-size: 12px; color: #5d6169;")
        detail_row.addWidget(self.lbl_speed)
        detail_row.addStretch()
        detail_row.addWidget(self.lbl_eta)
        card_layout.addLayout(detail_row)

        layout.addWidget(card)

        q_header = QLabel("실시간 파이프라인 로그")
        q_header.setStyleSheet("color: #8d929a; font-weight: 500; font-size: 12px;")
        layout.addWidget(q_header)

        self.log_list_widget = QListWidget()
        self.log_list_widget.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 11px;")
        layout.addWidget(self.log_list_widget)

        ctrl_layout = QHBoxLayout()
        btn_stop = QPushButton("작업 중지")
        btn_stop.setObjectName("SecondaryBtn")
        btn_stop.clicked.connect(self.stop_batch)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(btn_stop)
        layout.addLayout(ctrl_layout)

        self.stack.addWidget(w)

    # ── View 3: 작업 완료 ────────────────────────────────────────────────
    def setup_view3_completed(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        card = QFrame()
        card.setObjectName("CardPanel")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 28, 24, 24)
        card_layout.setSpacing(20)

        title = QLabel("모든 폴더 번역 작업이 완료되었습니다.")
        title.setStyleSheet("font-size: 16px; font-weight: 500; color: #ededed;")
        self.lbl_comp_sub = QLabel("총 0개 폴더 처리 완료")
        self.lbl_comp_sub.setStyleSheet("font-size: 13px; color: #8d929a;")
        card_layout.addWidget(title)
        card_layout.addWidget(self.lbl_comp_sub)

        stats_box = QHBoxLayout()
        stats_box.setSpacing(12)

        def make_stat_box(label_text, val_text, color_hex):
            f = QFrame()
            f.setStyleSheet("background-color: #1f2126; border: 1px solid #292b32; border-radius: 6px;")
            l = QVBoxLayout(f)
            l.setContentsMargins(14, 12, 14, 12)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 11px; color: #8d929a; border: none;")
            val = QLabel(val_text)
            val.setStyleSheet(f"font-size: 16px; font-weight: 500; color: {color_hex}; border: none;")
            l.addWidget(lbl)
            l.addWidget(val)
            return f, val

        f1, self.lbl_stat_success = make_stat_box("성공 완료", "0 pages", "#ededed")
        f2, self.lbl_stat_cache = make_stat_box("캐시 적중/스킵", "0 pages", "#4e82ee")
        f3, self.lbl_stat_error = make_stat_box("오류 발생", "0 pages", "#e35d6a")
        stats_box.addWidget(f1)
        stats_box.addWidget(f2)
        stats_box.addWidget(f3)
        card_layout.addLayout(stats_box)

        out_lbl = QLabel("완료된 폴더 및 생성된 결과물 경로")
        out_lbl.setStyleSheet("color: #8d929a; font-size: 12px; margin-top: 4px;")
        card_layout.addWidget(out_lbl)

        self.out_list_widget = QListWidget()
        card_layout.addWidget(self.out_list_widget)

        act_layout = QHBoxLayout()
        btn_open = QPushButton("출력 폴더 열기 (Finder)")
        btn_open.setObjectName("PrimaryBtn")
        btn_open.clicked.connect(self.open_output_folder)

        act_layout.addStretch()
        act_layout.addWidget(btn_open)
        card_layout.addLayout(act_layout)

        layout.addWidget(card)
        self.stack.addWidget(w)

    def switch_view(self, idx):
        self.stack.setCurrentIndex(idx)
        btns = [self.btn_tab1, self.btn_tab2, self.btn_tab3]
        for i, b in enumerate(btns):
            b.setProperty("class", "TabBtnActive" if i == idx else "TabBtn")
            b.style().unpolish(b)
            b.style().polish(b)

    def open_api_key_dialog(self):
        dlg = ApiKeyDialog(self.api_key, self)
        if dlg.exec_() == QDialog.Accepted:
            new_key = dlg.get_key()
            self.api_key = new_key
            save_env_api_key(new_key)
            self.check_system_health()
            QMessageBox.information(self, "저장 완료", "OpenRouter API Key가 .env 파일에 안전하게 저장되었습니다.")

    def check_system_health(self):
        lama_model = REPO_ROOT / "models" / "lama.onnx"
        if lama_model.exists():
            self.lama_status.setText("● Inpainting: LaMa ONNX [정상]")
            self.lama_status.setStyleSheet("color: #38c172;")
        else:
            self.lama_status.setText("● Inpainting: Flat Fill 기본 모드")
            self.lama_status.setStyleSheet("color: #8d929a;")

        if self.api_key and self.api_key.startswith("sk-or-"):
            self.api_status.setText("● OpenRouter API [설정됨]")
            self.api_status.setStyleSheet("color: #38c172;")
        elif self.api_key:
            self.api_status.setText("● OpenRouter API [키 감지됨]")
            self.api_status.setStyleSheet("color: #38c172;")
        else:
            self.api_status.setText("● OpenRouter API [미설정]")
            self.api_status.setStyleSheet("color: #e35d6a;")

    def add_folders(self, paths):
        for p in paths:
            if p not in self.registered_folders and os.path.isdir(p):
                self.registered_folders.append(p)
        self.update_folder_list()

    def clear_folders(self):
        self.registered_folders.clear()
        self.update_folder_list()

    def update_folder_list(self):
        self.folder_list_widget.clear()
        total_pages = 0
        for f in self.registered_folders:
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                imgs.extend(glob.glob(os.path.join(f, ext)))
            cnt = len(imgs)
            total_pages += cnt

            name = os.path.basename(f) or f
            item = QListWidgetItem(f"{name}   ({cnt} pages)\n  └ {f}")
            self.folder_list_widget.addItem(item)

        self.lbl_folder_count.setText(f"등록된 폴더 목록 ({len(self.registered_folders)}개)")
        self.lbl_page_count.setText(f"총 {total_pages} 페이지")

    def start_batch(self):
        if not self.api_key:
            reply = QMessageBox.warning(
                self,
                "API Key 필요",
                "OpenRouter API Key가 설정되지 않았습니다.\n지금 입력하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.open_api_key_dialog()
            if not self.api_key:
                return

        if not self.registered_folders:
            QMessageBox.warning(self, "안내", "번역할 만화 폴더를 먼저 드래그 앤 드롭으로 추가해주세요.")
            return

        self.switch_view(1)
        self.log_list_widget.clear()
        self.log_list_widget.addItem(QListWidgetItem("🚀 내부 백그라운드 스레드에서 파이프라인 구동 시작..."))

        self.worker = RealBatchWorkerThread(
            self.registered_folders,
            self.api_key,
            self.chk_skip_existing.isChecked(),
        )
        self.worker.progress_updated.connect(self.on_worker_progress)
        self.worker.log_emitted.connect(self.on_worker_log)
        self.worker.all_completed.connect(self.on_worker_all_done)
        self.worker.start()

    def stop_batch(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.lbl_cur_folder.setText("작업 중지됨")

    def on_worker_progress(self, folder, cur, total, speed, eta):
        pct = int((cur / total) * 100) if total > 0 else 0
        self.lbl_cur_folder.setText(f"{folder} 번역 중...")
        self.lbl_cur_ratio.setText(f"{cur} / {total} 페이지 ({pct}%)")
        self.progress_bar.setValue(pct)
        self.lbl_speed.setText(f"처리 속도: {speed} pages/min")
        self.lbl_eta.setText(f"예상 남은 시간: {eta}")

    def on_worker_log(self, msg):
        item = QListWidgetItem(msg)
        self.log_list_widget.addItem(item)
        self.log_list_widget.scrollToBottom()

    def on_worker_all_done(self, success, cache, err):
        self.switch_view(2)
        self.lbl_comp_sub.setText(f"총 {len(self.registered_folders)}개 폴더 처리 종료")
        self.lbl_stat_success.setText(f"{success} pages")
        self.lbl_stat_cache.setText(f"{cache} pages")
        self.lbl_stat_error.setText(f"{err} pages")

        self.out_list_widget.clear()
        out_base = REPO_ROOT / "output"
        for f in self.registered_folders:
            name = os.path.basename(f) or f
            folder_out = out_base / name
            out_imgs = glob.glob(str(folder_out / "*.png")) + glob.glob(str(folder_out / "*.jpg"))
            item = QListWidgetItem(f"📁 {name} ({len(out_imgs)}장 최종 생성)\n  └ {folder_out}")
            self.out_list_widget.addItem(item)

    def open_output_folder(self):
        out_dir = REPO_ROOT / "output"
        os.makedirs(out_dir, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))


def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_UTILITY_QSS)
    win = MangaTransDesktopApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
