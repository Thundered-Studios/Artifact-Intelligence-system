"""
app.py
------
AIS — Artifact Intelligence System
GUI for archaeologists: upload a photo, find similar artifacts.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import config
from ais.search import ArtifactSearcher

# ── Layout constants ───────────────────────────────────────────────────────────
WIN_W, WIN_H     = 960, 640
PREVIEW_SIZE     = 260          # uploaded image preview (px)
THUMB_SIZE       = 160          # result thumbnail (px)
RESULT_COLS      = 3
RESULT_ROWS      = 2
TOP_K            = RESULT_COLS * RESULT_ROWS

BG          = "#F5F2ED"         # warm off-white
PANEL_BG    = "#EAE6DF"
ACCENT      = "#5C4827"         # earthy brown
BTN_FG      = "#FFFFFF"
LABEL_FG    = "#2E2013"
MUTED_FG    = "#8A7560"
BORDER      = "#C8BEAF"
FONT_TITLE  = ("Helvetica", 18, "bold")
FONT_LABEL  = ("Helvetica", 10)
FONT_CLASS  = ("Helvetica", 11, "bold")
FONT_SCORE  = ("Helvetica", 9)
FONT_BTN    = ("Helvetica", 11, "bold")
FONT_STATUS = ("Helvetica", 9)


def _rounded_btn(parent, text, command, width=20, bg=ACCENT, fg=BTN_FG, state="normal"):
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=MUTED_FG, activeforeground=BTN_FG,
        font=FONT_BTN, relief="flat", bd=0, padx=12, pady=8,
        width=width, cursor="hand2", state=state,
    )


def _pil_to_tk(img: Image.Image, size: int) -> ImageTk.PhotoImage:
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    # Pad to square
    canvas = Image.new("RGB", (size, size), BG)
    x = (size - img.width)  // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return ImageTk.PhotoImage(canvas)


class AISApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AIS — Artifact Intelligence System")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._searcher: ArtifactSearcher | None = None
        self._uploaded_image: Image.Image | None = None
        self._thumb_refs: list[ImageTk.PhotoImage] = []   # prevent GC
        self._preview_ref: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._load_model_async()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=ACCENT, height=54)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        tk.Label(
            title_bar,
            text="  Artifact Intelligence System",
            bg=ACCENT, fg=BTN_FG, font=FONT_TITLE,
            anchor="w",
        ).pack(side="left", padx=16, pady=8)

        # ── Main area ─────────────────────────────────────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=16)

        self._build_left_panel(main)
        self._build_right_panel(main)

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = tk.Frame(self, bg=PANEL_BG, height=28)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Loading model…")
        tk.Label(
            status_bar, textvariable=self._status_var,
            bg=PANEL_BG, fg=MUTED_FG, font=FONT_STATUS, anchor="w",
        ).pack(side="left", padx=10)

    def _build_left_panel(self, parent: tk.Frame) -> None:
        left = tk.Frame(parent, bg=PANEL_BG, width=300, bd=1, relief="flat",
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)

        tk.Label(
            left, text="Upload Artifact Photo",
            bg=PANEL_BG, fg=LABEL_FG, font=("Helvetica", 12, "bold"),
        ).pack(pady=(20, 10))

        # Image preview area
        self._preview_label = tk.Label(
            left,
            bg=BG,
            width=PREVIEW_SIZE, height=PREVIEW_SIZE,
            relief="flat",
            highlightbackground=BORDER, highlightthickness=1,
            text="No image selected",
            fg=MUTED_FG, font=FONT_LABEL,
            cursor="hand2",
        )
        self._preview_label.pack(padx=16)
        self._preview_label.bind("<Button-1>", lambda _: self._upload())

        tk.Frame(left, bg=PANEL_BG, height=12).pack()

        self._upload_btn = _rounded_btn(left, "Choose Photo", self._upload, width=18)
        self._upload_btn.pack(pady=(0, 8))

        self._search_btn = _rounded_btn(
            left, "Search", self._search, width=18, state="disabled"
        )
        self._search_btn.pack()

        # Build-index button (secondary, smaller)
        tk.Frame(left, bg=PANEL_BG, height=16).pack()
        self._index_btn = tk.Button(
            left, text="Build / Rebuild Index",
            command=self._build_index,
            bg=PANEL_BG, fg=MUTED_FG,
            font=("Helvetica", 9), relief="flat", bd=0,
            cursor="hand2",
        )
        self._index_btn.pack()

    def _build_right_panel(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(
            right, text="Similar Artifacts",
            bg=BG, fg=LABEL_FG, font=("Helvetica", 13, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        # Grid container
        self._results_frame = tk.Frame(right, bg=BG)
        self._results_frame.pack(fill="both", expand=True)

        # Prediction banner (hidden until search)
        self._prediction_var = tk.StringVar()
        self._prediction_label = tk.Label(
            right, textvariable=self._prediction_var,
            bg=BG, fg=ACCENT, font=("Helvetica", 13, "bold"), anchor="w",
        )
        # Packed after a search

        self._show_placeholder()

    def _show_placeholder(self) -> None:
        for w in self._results_frame.winfo_children():
            w.destroy()
        tk.Label(
            self._results_frame,
            text="Upload a photo and press Search\nto find similar artifacts.",
            bg=BG, fg=MUTED_FG, font=("Helvetica", 12),
            justify="center",
        ).place(relx=0.5, rely=0.4, anchor="center")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model_async(self) -> None:
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        ckpt = Path(config.CHECKPOINT_DIR) / config.CHECKPOINT_NAME
        if not ckpt.exists():
            self._set_status(
                "No model found. Train the model first (python train.py)."
            )
            return
        try:
            self._searcher = ArtifactSearcher(
                checkpoint=ckpt,
                embedding_dim=config.EMBEDDING_DIM,
                num_classes=config.NUM_CLASSES,
                index_dir=config.CHECKPOINT_DIR,
            )
            if self._searcher.index_ready:
                n = len(self._searcher.image_paths)
                self._set_status(f"Ready — {n} artifacts indexed.")
                self._enable_search()
            else:
                self._set_status(
                    "Model loaded. No index yet — click 'Build / Rebuild Index'."
                )
        except Exception as exc:
            self._set_status(f"Error loading model: {exc}")

    # ── Upload ────────────────────────────────────────────────────────────────

    def _upload(self) -> None:
        path = filedialog.askopenfilename(
            title="Select artifact photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        img = Image.open(path).convert("RGB")
        self._uploaded_image = img

        ref = _pil_to_tk(img, PREVIEW_SIZE)
        self._preview_ref = ref
        self._preview_label.configure(image=ref, text="")

        if self._searcher and self._searcher.index_ready:
            self._enable_search()
        self._set_status("Photo loaded. Press Search.")

    # ── Search ────────────────────────────────────────────────────────────────

    def _enable_search(self) -> None:
        self._search_btn.configure(state="normal")

    def _search(self) -> None:
        if self._uploaded_image is None:
            messagebox.showinfo("AIS", "Please upload an artifact photo first.")
            return
        if self._searcher is None or not self._searcher.index_ready:
            messagebox.showinfo(
                "AIS", "Index not ready. Click 'Build / Rebuild Index' first."
            )
            return

        self._search_btn.configure(state="disabled", text="Searching…")
        self._set_status("Searching…")

        def _run():
            try:
                results = self._searcher.search(self._uploaded_image, top_k=TOP_K)
                self.after(0, lambda: self._show_results(results))
            except Exception as exc:
                self.after(0, lambda: self._set_status(f"Search error: {exc}"))
            finally:
                self.after(0, lambda: self._search_btn.configure(
                    state="normal", text="Search"
                ))

        threading.Thread(target=_run, daemon=True).start()

    def _show_results(self, results: list[dict]) -> None:
        for w in self._results_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        if not results:
            tk.Label(
                self._results_frame, text="No results found.",
                bg=BG, fg=MUTED_FG, font=FONT_LABEL,
            ).pack()
            return

        # Prediction banner
        top_class = results[0]["predicted"] or results[0]["class"]
        self._prediction_var.set(f"Most likely: {top_class.replace('_', ' ').title()}")
        self._prediction_label.pack(anchor="w", pady=(0, 10))

        # Results grid
        grid = tk.Frame(self._results_frame, bg=BG)
        grid.pack(fill="both", expand=True)

        for i, res in enumerate(results):
            row, col = divmod(i, RESULT_COLS)
            cell = tk.Frame(grid, bg=BG, padx=6, pady=6)
            cell.grid(row=row, column=col, sticky="nsew")

            # Thumbnail
            try:
                img = Image.open(res["path"]).convert("RGB")
                ref = _pil_to_tk(img, THUMB_SIZE)
            except Exception:
                ref = ImageTk.PhotoImage(Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), BORDER))
            self._thumb_refs.append(ref)

            thumb = tk.Label(
                cell, image=ref, bg=BG,
                highlightbackground=BORDER, highlightthickness=1,
            )
            thumb.pack()

            # Class label
            cls = res["class"].replace("_", " ").title()
            tk.Label(cell, text=cls, bg=BG, fg=LABEL_FG, font=FONT_CLASS).pack()

            # Similarity score
            pct = int(res["score"] * 100)
            tk.Label(
                cell, text=f"{pct}% match",
                bg=BG, fg=MUTED_FG, font=FONT_SCORE,
            ).pack()

        self._set_status(
            f"Found {len(results)} similar artifacts. "
            f"Best match: {top_class.replace('_', ' ').title()}"
        )

    # ── Index building ────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        if self._searcher is None:
            messagebox.showinfo("AIS", "Model not loaded yet.")
            return

        data_dir = Path(config.DATA_DIR)
        if not data_dir.exists():
            messagebox.showerror(
                "AIS",
                f"Data directory '{data_dir}' not found.\n"
                "Run the scraper first (python scrape.py).",
            )
            return

        self._index_btn.configure(state="disabled", text="Building…")
        self._set_status("Building index — this may take a moment…")

        progress_var = tk.StringVar(value="")
        prog_win = tk.Toplevel(self)
        prog_win.title("Building Index")
        prog_win.geometry("340x110")
        prog_win.resizable(False, False)
        prog_win.configure(bg=BG)
        prog_win.grab_set()

        tk.Label(prog_win, text="Indexing artifact images…",
                 bg=BG, fg=LABEL_FG, font=FONT_LABEL).pack(pady=(18, 6))
        bar = ttk.Progressbar(prog_win, length=280, mode="determinate")
        bar.pack()
        tk.Label(prog_win, textvariable=progress_var,
                 bg=BG, fg=MUTED_FG, font=FONT_STATUS).pack(pady=4)

        def _on_progress(current, total):
            pct = int(current / total * 100)
            self.after(0, lambda: bar.configure(value=pct))
            self.after(0, lambda: progress_var.set(f"{current} / {total}"))

        def _run():
            try:
                n = self._searcher.build_index(data_dir, on_progress=_on_progress)
                self.after(0, prog_win.destroy)
                self.after(0, lambda: self._set_status(f"Index built — {n} artifacts ready."))
                self.after(0, self._enable_search)
            except Exception as exc:
                self.after(0, prog_win.destroy)
                self.after(0, lambda: messagebox.showerror("AIS", str(exc)))
                self.after(0, lambda: self._set_status("Index build failed."))
            finally:
                self.after(0, lambda: self._index_btn.configure(
                    state="normal", text="Build / Rebuild Index"
                ))

        threading.Thread(target=_run, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self.after(0, lambda: self._status_var.set(f"  {msg}"))


if __name__ == "__main__":
    app = AISApp()
    app.mainloop()
