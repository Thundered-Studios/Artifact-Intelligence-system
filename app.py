"""
app.py
------
AIS — Artifact Intelligence System
Upload an artifact photo + optional text description → Analyze → results.

Features:
  - Multimodal: image + text description combined for better accuracy
  - Deep research: academic article search via OpenAlex (250M+ papers)
  - Corrections: archaeologists can flag wrong predictions to improve the model
"""

from __future__ import annotations

import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import config
from ais.data.scraper import scrape_dataset
from ais.feedback import FeedbackStore
from ais.research import build_query, search_articles
from ais.search import ArtifactSearcher

WIN_W, WIN_H = 960, 700
PREVIEW_SIZE = 220
THUMB_SIZE   = 130
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

        self._searcher:       ArtifactSearcher | None = None
        self._feedback:       FeedbackStore           = FeedbackStore(config.CHECKPOINT_DIR)
        self._uploaded_image: Image.Image | None      = None
        self._last_embedding: object                  = None   # torch.Tensor
        self._last_predicted: str                     = ""
        self._thumb_refs:     list[ImageTk.PhotoImage] = []
        self._preview_ref:    ImageTk.PhotoImage | None = None
        self._article_urls:   list[str]               = []

        self._build_ui()
        self._load_searcher_async()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        menubar = tk.Menu(self)
        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Rebuild Reference Database", command=self._confirm_rebuild)
        tools.add_separator()
        tools.add_command(
            label="Refine Model (trains on current data)", command=self._refine_model
        )
        menubar.add_cascade(label="Tools", menu=tools)
        self.configure(menu=menubar)

        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=8)

        self._build_left(main)
        tk.Frame(main, width=1, bg="gray70").pack(side="left", fill="y", padx=8)
        self._build_right(main)

        self._status_var = tk.StringVar(value="Starting up...")
        tk.Label(
            self, textvariable=self._status_var,
            relief="sunken", anchor="w", bd=1,
        ).pack(fill="x", side="bottom", ipady=2)

    def _build_left(self, parent: tk.Frame) -> None:
        left = tk.Frame(parent, width=270)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="Artifact Photo").pack(pady=(4, 4))

        self._preview_label = tk.Label(
            left,
            width=PREVIEW_SIZE, height=PREVIEW_SIZE,
            relief="sunken", bd=2,
            text="(click to choose photo)",
            fg="gray50", wraplength=180, cursor="hand2",
        )
        self._preview_label.pack()
        self._preview_label.bind("<Button-1>", lambda _: self._upload())

        tk.Frame(left, height=6).pack()
        tk.Button(left, text="Choose Photo...", command=self._upload, width=20).pack()

        tk.Frame(left, height=8).pack()
        tk.Label(left, text="Description (optional):", anchor="w").pack(fill="x", padx=4)
        self._text_input = tk.Text(left, height=4, width=26, wrap="word", relief="sunken", bd=1)
        self._text_input.pack(padx=4)
        tk.Label(
            left,
            text="e.g. 'bronze coin with seated figure'",
            fg="gray50", font=("TkDefaultFont", 8),
            wraplength=240, justify="left",
        ).pack(anchor="w", padx=4)

        tk.Frame(left, height=8).pack()
        self._analyze_btn = tk.Button(
            left, text="Analyze", command=self._analyze,
            width=20, state="disabled",
        )
        self._analyze_btn.pack()

        # Correction UI — shown after a result
        tk.Frame(left, height=12).pack()
        tk.Frame(left, height=1, bg="gray70").pack(fill="x", padx=4)
        tk.Label(left, text="Was the prediction correct?", fg="gray50",
                 font=("TkDefaultFont", 8)).pack(pady=(6, 2))

        btn_row = tk.Frame(left)
        btn_row.pack()
        tk.Button(btn_row, text="Yes", width=8, command=self._correct_yes).pack(side="left", padx=2)
        tk.Button(btn_row, text="No", width=8, command=self._correct_no).pack(side="left", padx=2)

        tk.Label(left, text="Correct class:", fg="gray50",
                 font=("TkDefaultFont", 8)).pack(pady=(6, 0))
        self._correction_var = tk.StringVar()
        self._correction_cb  = ttk.Combobox(
            left, textvariable=self._correction_var,
            state="disabled", width=20,
        )
        self._correction_cb.pack(padx=4)
        self._submit_btn = tk.Button(
            left, text="Submit Correction", width=20,
            command=self._submit_correction, state="disabled",
        )
        self._submit_btn.pack(pady=(4, 0))

    def _build_right(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        # ── Top: image results ────────────────────────────────────────────────
        tk.Label(right, text="Similar Artifacts", anchor="w").pack(fill="x")
        tk.Frame(right, height=1, bg="gray70").pack(fill="x", pady=(2, 4))

        self._prediction_var = tk.StringVar()
        tk.Label(right, textvariable=self._prediction_var, anchor="w").pack(fill="x")

        self._results_frame = tk.Frame(right)
        self._results_frame.pack(fill="x")
        self._show_placeholder()

        # ── Bottom: articles ──────────────────────────────────────────────────
        tk.Frame(right, height=1, bg="gray70").pack(fill="x", pady=(8, 4))
        tk.Label(right, text="Related Academic Articles", anchor="w").pack(fill="x")

        self._articles_frame = tk.Frame(right)
        self._articles_frame.pack(fill="both", expand=True, pady=(4, 0))

        tk.Label(
            self._articles_frame,
            text="Articles will appear here after analysis.",
            fg="gray50",
        ).pack(anchor="w")

    def _show_placeholder(self) -> None:
        for w in self._results_frame.winfo_children():
            w.destroy()
        tk.Label(
            self._results_frame,
            text="Upload a photo and click Analyze.",
            fg="gray50",
        ).pack(pady=20)

    # ── Startup ───────────────────────────────────────────────────────────────

    def _load_searcher_async(self) -> None:
        threading.Thread(target=self._load_searcher, daemon=True).start()

    def _load_searcher(self) -> None:
        try:
            self._set_status("Loading model...")
            self._searcher = ArtifactSearcher(
                embedding_dim=config.EMBEDDING_DIM,
                index_dir=config.CHECKPOINT_DIR,
            )

            # Try pulling latest embeddings from Firebase cloud database
            from ais.firebase_client import get_client
            client = get_client()
            if client and client.connected:
                self._set_status("Connected to cloud database — syncing...")
                version = client.get_current_version() or "v1"
                counts  = client.get_artifact_counts()
                total_cloud = sum(counts.values())
                if total_cloud > 0:
                    added = self._searcher.pull_from_cloud(
                        version=version,
                        classes=list(counts.keys()),
                        cache_dir=Path(config.DATA_DIR),
                        on_progress=self._set_status,
                    )
                    if added:
                        self._set_status(
                            f"Cloud sync: +{added} artifacts from shared database."
                        )

            if self._searcher.index_ready:
                n = len(self._searcher.image_paths)
                self._set_status(f"Ready — {n} artifacts in reference database.")
            else:
                self._set_status("Ready. Upload a photo and click Analyze.")
            self.after(0, self._refresh_analyze_btn)
        except Exception as exc:
            self._set_status(f"Startup error: {exc}")

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

        self._refresh_analyze_btn()
        self._set_status("Photo loaded. Click Analyze.")

    def _refresh_analyze_btn(self) -> None:
        ready = self._uploaded_image is not None and self._searcher is not None
        self._analyze_btn.configure(state="normal" if ready else "disabled")

    # ── Analyze ───────────────────────────────────────────────────────────────

    def _analyze(self) -> None:
        if self._uploaded_image is None or self._searcher is None:
            return
        self._analyze_btn.configure(state="disabled", text="Working...")

        if not self._searcher.index_ready:
            threading.Thread(target=self._first_run_then_search, daemon=True).start()
        else:
            threading.Thread(target=self._run_search, daemon=True).start()

    def _first_run_then_search(self) -> None:
        try:
            data_dir = Path(config.DATA_DIR)
            self._set_status("First run: downloading reference images (~2 min)...")
            scrape_dataset(
                classes=config.QUICK_CLASSES,
                out_dir=data_dir,
                n_images=config.QUICK_N_IMAGES,
                val_split=0.2,
                on_progress=self._set_status,
            )
            self._set_status("Building search index...")
            n = self._searcher.build_index(data_dir, on_progress=self._set_status)
            self._set_status(f"Index built — {n} reference artifacts.")

            # Push to cloud so other users benefit from this scrape
            from ais.firebase_client import get_client
            client = get_client()
            if client and client.connected:
                self._set_status("Uploading artifacts to cloud database...")
                self._searcher.push_to_cloud(
                    data_dir=data_dir,
                    version="v1",
                    on_progress=self._set_status,
                )

            self._run_search()
        except Exception as exc:
            self._set_status(f"Setup failed: {exc}")
            self.after(0, lambda: self._analyze_btn.configure(
                state="normal", text="Analyze"
            ))

    def _run_search(self) -> None:
        try:
            text = self._text_input.get("1.0", "end").strip()
            self._set_status("Analyzing...")
            results, embedding = self._searcher.search(
                self._uploaded_image, text=text, top_k=TOP_K
            )
            self._last_embedding = embedding
            self._last_predicted = results[0]["predicted"] if results else ""
            self.after(0, lambda: self._show_results(results))

            # Article search runs after visual results are shown
            predicted = self._last_predicted
            threading.Thread(
                target=self._fetch_articles,
                args=(predicted, text),
                daemon=True,
            ).start()

        except Exception as exc:
            self._set_status(f"Analysis failed: {exc}")
        finally:
            self.after(0, lambda: self._analyze_btn.configure(
                state="normal", text="Analyze"
            ))

    # ── Results ───────────────────────────────────────────────────────────────

    def _show_results(self, results: list[dict]) -> None:
        for w in self._results_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        if not results:
            tk.Label(self._results_frame, text="No results found.").pack()
            return

        top_class  = results[0]["predicted"]
        confidence = results[0]["confidence"]
        self._prediction_var.set(
            f"Most likely: {top_class.replace('_', ' ').title()}    ({confidence})"
        )

        # Populate correction combobox with known classes
        known = sorted(set(r["class"] for r in results))
        if self._searcher and self._searcher.class_names:
            known = sorted(set(self._searcher.class_names))
        self._correction_cb.configure(values=known, state="readonly")
        self._correction_var.set("")

        grid = tk.Frame(self._results_frame)
        grid.pack()

        for i, res in enumerate(results):
            row, col = divmod(i, RESULT_COLS)
            cell = tk.Frame(grid, bd=1, relief="sunken", padx=3, pady=3)
            cell.grid(row=row, column=col, padx=5, pady=5)

            try:
                img = Image.open(res["path"]).convert("RGB")
                ref = _pil_to_tk(img, THUMB_SIZE)
            except Exception:
                ref = ImageTk.PhotoImage(
                    Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), "#d9d9d9")
                )
            self._thumb_refs.append(ref)

            tk.Label(cell, image=ref).pack()
            tk.Label(cell, text=res["class"].replace("_", " ").title()).pack()
            pct = max(0, int(res["score"] * 100))
            tk.Label(cell, text=f"{pct}% match", fg="gray50").pack()

        self._set_status(
            f"Done — most likely {top_class.replace('_', ' ').title()}. "
            f"{len(results)} similar artifacts shown."
        )

    # ── Articles ──────────────────────────────────────────────────────────────

    def _fetch_articles(self, predicted_class: str, user_text: str) -> None:
        self.after(0, self._clear_articles)
        self._set_status("Searching academic literature...")
        query    = build_query(predicted_class, user_text)
        articles = search_articles(query, max_results=4)
        self.after(0, lambda: self._show_articles(articles))

    def _clear_articles(self) -> None:
        for w in self._articles_frame.winfo_children():
            w.destroy()

    def _show_articles(self, articles: list[dict]) -> None:
        self._clear_articles()
        self._article_urls = []

        if not articles:
            tk.Label(self._articles_frame, text="No articles found.", fg="gray50").pack(anchor="w")
            self._set_status("Done.")
            return

        for i, art in enumerate(articles):
            row = tk.Frame(self._articles_frame)
            row.pack(fill="x", pady=2)

            year  = f" ({art['year']})" if art["year"] else ""
            cited = f"  [{art['cited_by']} citations]" if art["cited_by"] else ""
            title = art["title"]
            if len(title) > 90:
                title = title[:87] + "..."

            label_text = f"• {title}{year}{cited}"
            lbl = tk.Label(
                row, text=label_text, anchor="w",
                fg="blue" if art["url"] else "black",
                cursor="hand2" if art["url"] else "",
                wraplength=630, justify="left",
            )
            lbl.pack(anchor="w")

            if art["url"]:
                url = art["url"]
                lbl.bind("<Button-1>", lambda _e, u=url: webbrowser.open(u))

            self._article_urls.append(art["url"])

        self._set_status(
            f"Done — {len(articles)} related articles found. "
            "Click article title to open."
        )

    # ── Correction UI ─────────────────────────────────────────────────────────

    def _correct_yes(self) -> None:
        if not self._last_predicted:
            return
        self._set_status("Confirmed — prediction recorded as correct.")

    def _correct_no(self) -> None:
        self._correction_cb.configure(state="readonly")
        self._submit_btn.configure(state="normal")
        self._set_status("Select the correct class from the dropdown, then click Submit.")

    def _submit_correction(self) -> None:
        correct = self._correction_var.get().strip()
        if not correct:
            messagebox.showinfo("AIS", "Please select the correct class first.")
            return
        if self._last_embedding is None:
            messagebox.showinfo("AIS", "Please run an analysis first.")
            return

        self._feedback.add(
            embedding=self._last_embedding,
            predicted=self._last_predicted,
            correct=correct,
        )
        self._correction_cb.configure(state="disabled")
        self._submit_btn.configure(state="disabled")
        self._set_status(
            f"Correction saved ({len(self._feedback)} total). "
            + ("Consider using Tools → Refine Model to improve accuracy."
               if self._feedback.should_suggest_retrain else "")
        )

        if self._feedback.should_suggest_retrain:
            self.after(500, self._suggest_retrain)

    def _suggest_retrain(self) -> None:
        if messagebox.askyesno(
            "Improve Accuracy",
            f"You have submitted {len(self._feedback)} corrections.\n"
            "Retrain the model now to improve accuracy?",
        ):
            self._refine_model()

    # ── Rebuild / Refine ──────────────────────────────────────────────────────

    def _confirm_rebuild(self) -> None:
        if messagebox.askyesno(
            "Rebuild Reference Database",
            "This will re-download all reference images and rebuild the index.\nContinue?",
        ):
            if self._searcher:
                if self._searcher.index_path.exists():
                    self._searcher.index_path.unlink()
                self._searcher.embeddings  = None
                self._searcher.image_paths = []
                self._searcher.class_names = []
            self._set_status("Index cleared. Click Analyze to rebuild.")

    def _refine_model(self) -> None:
        if self._searcher is None:
            messagebox.showinfo("AIS", "Model not loaded yet.")
            return
        data_dir = Path(config.DATA_DIR)
        if not (data_dir / "train").exists():
            messagebox.showerror("AIS", "No training data found. Run Analyze first.")
            return
        if not messagebox.askyesno(
            "Refine Model",
            "Train the model on your current reference images (~3 min on CPU).\n"
            "This improves accuracy for your specific artifact collection.\nContinue?",
        ):
            return

        def _run():
            try:
                self._set_status("Refining model...")
                self._searcher.train(
                    data_dir=data_dir,
                    epochs=config.EPOCHS,
                    batch_size=config.BATCH_SIZE,
                    lr=config.LEARNING_RATE,
                    on_progress=self._set_status,
                )
                self._set_status("Rebuilding index with refined model...")
                n = self._searcher.build_index(data_dir, on_progress=self._set_status)
                self._feedback.clear()
                self.after(0, self._refresh_analyze_btn)
                self._set_status(f"Model refined — {n} artifacts re-indexed.")
            except Exception as exc:
                self._set_status(f"Refinement failed: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enable_search(self) -> None:
        self._analyze_btn.configure(state="normal")

    def _set_status(self, msg: str) -> None:
        self.after(0, lambda: self._status_var.set(f" {msg}"))


if __name__ == "__main__":
    app = AISApp()
    app.mainloop()
