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

WIN_W, WIN_H = 960, 640
PREVIEW_SIZE = 260
THUMB_SIZE   = 150
RESULT_COLS  = 3
RESULT_ROWS  = 2
TOP_K        = RESULT_COLS * RESULT_ROWS


def _pil_to_tk(img: Image.Image, size: int) -> ImageTk.PhotoImage:
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), "#d9d9d9")
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

        self._searcher: ArtifactSearcher | None = None
        self._uploaded_image: Image.Image | None = None
        self._thumb_refs: list[ImageTk.PhotoImage] = []
        self._preview_ref: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._load_model_async()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Menu bar ──────────────────────────────────────────────────────────
        menubar = tk.Menu(self)
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Build / Rebuild Index", command=self._build_index)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        self.configure(menu=menubar)

        # ── Main area ─────────────────────────────────────────────────────────
        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=8)

        self._build_left_panel(main)
        tk.Frame(main, width=1, bg="gray70").pack(side="left", fill="y", padx=8)
        self._build_right_panel(main)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Loading model...")
        tk.Label(
            self, textvariable=self._status_var,
            relief="sunken", anchor="w", bd=1,
        ).pack(fill="x", side="bottom", ipady=2)

    def _build_left_panel(self, parent: tk.Frame) -> None:
        left = tk.Frame(parent, width=290)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="Artifact Photo").pack(pady=(6, 4))

        self._preview_label = tk.Label(
            left,
            width=PREVIEW_SIZE, height=PREVIEW_SIZE,
            relief="sunken", bd=2,
            text="(no image)",
            cursor="hand2",
        )
        self._preview_label.pack()
        self._preview_label.bind("<Button-1>", lambda _: self._upload())

        tk.Frame(left, height=8).pack()
        tk.Button(left, text="Choose Photo...", command=self._upload, width=20).pack()
        tk.Frame(left, height=4).pack()
        self._search_btn = tk.Button(
            left, text="Search", command=self._search, width=20, state="disabled"
        )
        self._search_btn.pack()

    def _build_right_panel(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        self._prediction_var = tk.StringVar()
        self._prediction_label = tk.Label(
            right, textvariable=self._prediction_var, anchor="w"
        )
        # packed after first search

        tk.Label(right, text="Similar Artifacts", anchor="w").pack(fill="x")
        tk.Frame(right, height=1, bg="gray70").pack(fill="x", pady=(2, 6))

        self._results_frame = tk.Frame(right)
        self._results_frame.pack(fill="both", expand=True)

        self._show_placeholder()

    def _show_placeholder(self) -> None:
        for w in self._results_frame.winfo_children():
            w.destroy()
        tk.Label(
            self._results_frame,
            text="Upload a photo and press Search.",
            fg="gray50",
        ).place(relx=0.5, rely=0.4, anchor="center")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model_async(self) -> None:
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        ckpt = Path(config.CHECKPOINT_DIR) / config.CHECKPOINT_NAME
        if not ckpt.exists():
            self._set_status("No model found. Run train.py first.")
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
                self.after(0, self._enable_search)
            else:
                self._set_status("Model loaded. Use Tools > Build Index before searching.")
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
            messagebox.showinfo("AIS", "Please upload a photo first.")
            return
        if self._searcher is None or not self._searcher.index_ready:
            messagebox.showinfo("AIS", "Index not ready. Use Tools > Build Index.")
            return

        self._search_btn.configure(state="disabled", text="Searching...")
        self._set_status("Searching...")

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
            tk.Label(self._results_frame, text="No results found.").pack()
            return

        top_class = results[0]["predicted"] or results[0]["class"]
        self._prediction_var.set(f"Most likely: {top_class.replace('_', ' ').title()}")
        self._prediction_label.pack(fill="x", pady=(0, 6))

        grid = tk.Frame(self._results_frame)
        grid.pack(fill="both", expand=True)

        for i, res in enumerate(results):
            row, col = divmod(i, RESULT_COLS)
            cell = tk.Frame(grid, bd=1, relief="sunken", padx=4, pady=4)
            cell.grid(row=row, column=col, padx=6, pady=6)

            try:
                img = Image.open(res["path"]).convert("RGB")
                ref = _pil_to_tk(img, THUMB_SIZE)
            except Exception:
                ref = ImageTk.PhotoImage(Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), "#d9d9d9"))
            self._thumb_refs.append(ref)

            tk.Label(cell, image=ref).pack()
            tk.Label(cell, text=res["class"].replace("_", " ").title()).pack()
            tk.Label(cell, text=f"{int(res['score'] * 100)}% match", fg="gray50").pack()

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
                "Run scrape.py first.",
            )
            return

        self._set_status("Building index...")

        progress_var = tk.StringVar(value="")
        prog_win = tk.Toplevel(self)
        prog_win.title("Building Index")
        prog_win.geometry("320x100")
        prog_win.resizable(False, False)
        prog_win.grab_set()

        tk.Label(prog_win, text="Indexing artifact images...").pack(pady=(14, 4))
        bar = ttk.Progressbar(prog_win, length=260, mode="determinate")
        bar.pack()
        tk.Label(prog_win, textvariable=progress_var, fg="gray50").pack(pady=4)

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

        threading.Thread(target=_run, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self.after(0, lambda: self._status_var.set(f" {msg}"))


if __name__ == "__main__":
    app = AISApp()
    app.mainloop()
