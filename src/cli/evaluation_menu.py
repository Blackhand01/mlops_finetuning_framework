import json
from datetime import datetime
from pathlib import Path
from .base_menu import BaseMenu
from config import ConfigLoader


class EvaluationMenu(BaseMenu):
    """Interactive menu for the OpenAI Evaluation API."""

    def __init__(self, eval_manager, file_manager):
        """
        Args:
            eval_manager: instance of EvaluationManager
            file_manager: instance of OpenAIFileManager (to upload files for runs)
        """
        self.eval_manager = eval_manager
        self.file_manager = file_manager

        # path a src/config.yaml
        config_path = Path(__file__).parent.parent / "config.yaml"
        src_root = config_path.parent

        try:
            # carico config utente
            cfg = ConfigLoader(str(config_path)).load()
            self.ev_cfg = cfg["evaluation"]
            dataset_base = cfg["dataset"]["base_dir"]

            # noto se è zero-config (autobuild già presente) o YAML-mode
            if "_autobuild" in cfg:
                self.autobuild = cfg["_autobuild"]
            else:
                # genero in runtime l'autobuild solo per reperire i file
                try:
                    auto_cfg = ConfigLoader(dataset_base).load()
                    self.autobuild = auto_cfg.get("_autobuild", {})
                except Exception:
                    self.autobuild = {}

            # risolvo i template dei path relativi nella sezione evaluation
            for key in ("data_source_config_path", "testing_criteria_path", "data_source_run_path"):
                self.ev_cfg[key] = str((src_root / self.ev_cfg[key]).resolve())

            self.auto = True
        except Exception as e:
            print(f"⚠️ Impossibile caricare {config_path}: {e}")
            print("    Useremo input manuale per evaluation")
            self.ev_cfg = {}
            self.autobuild = {}
            self.auto = False

    def show(self):
        """Display the Evaluation menu and process user choices."""
        while True:
            print("\n--- Evaluation Menu ---")
            print("1. Create evaluation + run")
            print("2. Retrieve by ID (evaluation/run/output)")
            print("3. List evaluations & runs")
            print("4. Update evaluation")
            print("5. Delete evaluation or run")
            print("6. Cancel evaluation run")
            print("7. Download run results")
            print("0. Return to main menu")
            choice = input("Select an option (0-7): ").strip()

            if choice == "1":
                self._create_evaluation_and_run()
            elif choice == "2":
                self._retrieve_by_id()
            elif choice == "3":
                self._list_evaluations_and_runs()
            elif choice == "4":
                self._update_evaluation()
            elif choice == "5":
                self._delete_evaluation_or_run()
            elif choice == "6":
                self._cancel_run()
            elif choice == "7":
                self._download_run_results()
            elif choice == "0":
                break
            else:
                print("Opzione non valida. Riprova.")

    def _create_evaluation_and_run(self):
        """Create an evaluation and immediately start a run."""
        name = input("Enter evaluation name: ").strip()

        # === Caricamento config evaluation ===
        if self.auto:
            try:
                with open(self.ev_cfg["data_source_config_path"], encoding="utf-8") as f:
                    data_src_cfg = json.load(f)
                with open(self.ev_cfg["testing_criteria_path"], encoding="utf-8") as f:
                    test_crit = json.load(f)
            except FileNotFoundError as e:
                print(f"❌ Config file non trovato: {e.filename}")
                return
        else:
            cfg_path  = input("Path to data_source_config.json: ").strip()
            crit_path = input("Path to testing_criteria.json: ").strip()
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    data_src_cfg = json.load(f)
                with open(crit_path, encoding="utf-8") as f:
                    test_crit = json.load(f)
            except Exception as e:
                print(f"❌ Errore caricando config manuale: {e}")
                return

        # === Creazione evaluation ===
        try:
            ev = self.eval_manager.create_evaluation(
                name=name,
                data_source_config=data_src_cfg,
                testing_criteria=test_crit,
                metadata={},
            )
            eid = ev.get("id")
            print(f"✅ Evaluation creata: ID={eid}")
        except Exception as e:
            print(f"❌ Errore creando evaluation: {e}")
            return

        # === Selezione automatica del file di eval ===
        if self.auto and "data" in self.autobuild and "eval" in self.autobuild["data"]:
            file_path = self.autobuild["data"]["eval"]
            print(f"Usando in automatico il file di eval: {file_path}")
        else:
            file_path = input("Enter local file path for evaluation run: ").strip()

        # === Upload del file JSONL da valutare ===
        try:
            up = self.file_manager.upload_file(
                path=file_path,
                purpose="evals",
                check_jsonl=True
            )
            file_id = up[0]["id"] if isinstance(up, list) else up["id"]
        except Exception as e:
            print(f"❌ Errore caricando file: {e}")
            return

        # === Creazione del run usando il template data_source_run.json ===
        if self.auto:
            try:
                with open(self.ev_cfg["data_source_run_path"], encoding="utf-8") as f:
                    run_cfg = json.load(f)
            except FileNotFoundError:
                print(f"❌ Run template non trovato in {self.ev_cfg['data_source_run_path']}")
                return

            # Sanitizzazione: se ci sono riferimenti "item.", rimuovili
            if "references" in run_cfg and isinstance(run_cfg["references"], str):
                if run_cfg["references"].startswith("item."):
                    run_cfg["references"] = run_cfg["references"].split(".", 1)[1]
            if "input_messages" in run_cfg:
                im = run_cfg["input_messages"]
                if im.get("type") == "item_reference" and isinstance(im.get("item_reference"), str):
                    if im["item_reference"].startswith("item."):
                        im["item_reference"] = im["item_reference"].split(".", 1)[1]

            run_cfg["source"]["id"] = file_id
            try:
                run = self.eval_manager.create_evaluation_run(eid, run_cfg)
                print(f"✅ Run creata (auto): ID={run.get('id')}")
            except Exception as e:
                print(f"❌ Errore creando run auto: {e}")
        else:
            data_source = {"source": {"type": "file_id", "id": file_id}}
            try:
                run = self.eval_manager.create_evaluation_run(eid, data_source)
                print(f"✅ Run creata (manuale): ID={run.get('id')}")
            except Exception as e:
                print(f"❌ Errore creando run manuale: {e}")

    def _retrieve_by_id(self):
        """Retrieve an evaluation, run or output-item by its ID prefix."""
        uid = input("Enter Evaluation / Run / Output ID: ").strip()
        try:
            if uid.startswith("outputitem_"):
                for ev in self.eval_manager.list_evaluations():
                    for run in self.eval_manager.list_evaluation_runs(ev["id"]):
                        try:
                            item = self.eval_manager.retrieve_output_item(ev["id"], run["id"], uid)
                            print(json.dumps(item, indent=2, ensure_ascii=False))
                            return
                        except:
                            pass
                print("❌ Output-item ID non trovato.")
            elif uid.startswith("evalrun_"):
                for ev in self.eval_manager.list_evaluations():
                    for run in self.eval_manager.list_evaluation_runs(ev["id"]):
                        if run["id"] == uid:
                            resp = self.eval_manager.retrieve_evaluation_run(uid, eval_id=ev["id"])
                            print(json.dumps(resp, indent=2, ensure_ascii=False))
                            return
                print("❌ Run ID non trovato.")
            elif uid.startswith("eval_"):
                ev = self.eval_manager.retrieve_evaluation(uid)
                runs = self.eval_manager.list_evaluation_runs(uid)
                print(json.dumps({"evaluation": ev, "runs": runs}, indent=2, ensure_ascii=False))
            else:
                print("❌ ID non valido. Deve iniziare con 'eval_', 'evalrun_' o 'outputitem_'.")
        except Exception as e:
            print(f"❌ Errore nel recupero: {e}")

    def _list_evaluations_and_runs(self):
        """List all evaluations and their runs with basic details."""
        try:
            evals = self.eval_manager.list_evaluations()
            if not evals:
                print("🔎 Nessuna evaluation trovata.")
                return
            for ev in evals:
                created = datetime.fromtimestamp(ev["created_at"]).strftime("%Y-%m-%d")
                print(f"\n🧪 {ev['id']}  |  {ev.get('name','')}  |  {created}")
                for r in self.eval_manager.list_evaluation_runs(ev["id"]):
                    ts     = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
                    passed = r.get("result_counts",{}).get("passed", 0)
                    total  = r.get("result_counts",{}).get("total", 0)
                    print(f"   • {r['id']} | {r['status']} | {ts} | passed {passed}/{total}")
        except Exception as e:
            print(f"❌ Errore nel listing: {e}")

    def _update_evaluation(self):
        """Update name and/or metadata of an existing evaluation."""
        eid      = input("Evaluation ID: ").strip()
        new_name = input("New name (blank to skip): ").strip() or None
        new_meta = input("New metadata JSON (blank to skip): ").strip()
        meta     = json.loads(new_meta) if new_meta else None
        try:
            upd = self.eval_manager.update_evaluation(eid, new_name, meta)
            print("✅ Updated:\n", json.dumps(upd, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ Errore updating evaluation: {e}")

    def _delete_evaluation_or_run(self):
        """Delete an evaluation or a run by ID."""
        uid = input("Evaluation or Run ID to delete: ").strip()
        try:
            if uid.startswith("eval_"):
                ok = self.eval_manager.delete_evaluation(uid)
                print(f"🗑️ Evaluation deleted: {ok}")
            elif uid.startswith("evalrun_"):
                for ev in self.eval_manager.list_evaluations():
                    try:
                        ok = self.eval_manager.delete_evaluation_run(uid, ev["id"])
                        if ok:
                            print(f"🗑️ Run {uid} deleted from eval {ev['id']}")
                            return
                    except:
                        pass
                print("❌ Run ID non trovato.")
            else:
                print("❌ ID non valido.")
        except Exception as e:
            print(f"❌ Errore nel delete: {e}")

    def _cancel_run(self):
        """Cancel an ongoing evaluation run."""
        rid = input("Run ID: ").strip()
        eid = input("Evaluation ID: ").strip()
        if input(f"Cancel run {rid}? (y/N): ").strip().lower() == "y":
            try:
                res = self.eval_manager.cancel_evaluation_run(rid, eid)
                print(f"⛔ Run cancelled: {res.get('status')}")
            except Exception as e:
                print(f"❌ Errore cancelling run: {e}")

    def _download_run_results(self):
        """Download output-items and evaluation metadata for offline analysis."""
        run_id = input("Enter Run ID: ").strip()

        # Trova l'eval parent
        eval_id = None
        for ev in self.eval_manager.list_evaluations():
            if any(r["id"] == run_id for r in self.eval_manager.list_evaluation_runs(ev["id"])):
                eval_id = ev["id"]
                break

        if not eval_id:
            print(f"❌ Run ID {run_id} non trovato.")
            return

        base_dir = Path("result") / "evals" / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # 1) Download output-items
        try:
            items = self.eval_manager.list_output_items(eval_id, run_id)
            if not items:
                print("⚠️ Nessun output-item per questa run.")
            else:
                results_path = base_dir / f"results_{run_id}.jsonl"
                with results_path.open("w", encoding="utf-8") as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"✅ Scaricati {len(items)} output-item in '{results_path}'.")
        except Exception as e:
            print(f"❌ Errore scaricando risultati: {e}")
            return

        # 2) Download evaluation metadata
        try:
            eval_meta = self.eval_manager.retrieve_evaluation(eval_id)
            metadata_path = base_dir / f"metadata_{run_id}.json"
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(eval_meta, f, ensure_ascii=False, indent=2)
            print(f"✅ Salvati metadata evaluation in '{metadata_path}'.")
        except Exception as e:
            print(f"❌ Errore scaricando metadata: {e}")
