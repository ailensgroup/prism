import datetime
import json
from pathlib import Path


class CheckpointManager:
    """Manages checkpointing and recovery for the evaluation pipeline.

    Provides functionality to create, save, load, and manage checkpoints during
    long-running evaluation processes. Supports resumable execution and result
    caching to prevent data loss during interruptions.

    Attributes:
        checkpoint_dir (Path): Directory where checkpoint files are stored.
        checkpoint_file (Path | None): Path to the current active checkpoint file.
        processed_ids (set): Set of query IDs that have been processed.
        results_cache (list): In-memory cache of all processed results.
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints") -> None:
        """Initialize the CheckpointManager with specified directory.

        Args:
            checkpoint_dir (str, optional): Directory path for storing checkpoint files.
                Defaults to "./checkpoints".
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_file = None
        self.processed_ids = set()
        self.results_cache = []

    def initialize_checkpoint(self, run_id: str, evaluation_type: str) -> Path:
        """Initialize a new checkpoint file for this evaluation run.

        Creates a new checkpoint file with metadata and prepares the manager
        for tracking progress during the evaluation process.

        Args:
            run_id (str): Unique identifier for this evaluation run.
            evaluation_type (str): Type of evaluation (e.g., "chunk_ranking",
                "document_ranking").

        Returns:
            Path: Path to the created checkpoint file.

        Raises:
            Exception: If checkpoint file creation fails.
        """
        try:
            # Create checkpoint directory if it doesn't exist
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Checkpoint directory: {self.checkpoint_dir}")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_file = self.checkpoint_dir / f"{evaluation_type}_{run_id}_{timestamp}.checkpoint"

            # Create initial checkpoint metadata
            metadata = {
                "run_id": run_id,
                "evaluation_type": evaluation_type,
                "started_at": timestamp,
                "total_processed": 0,
                "processed_ids": [],
                "results": [],
            }

            print(f"📝 Writing checkpoint file: {self.checkpoint_file}")
            with open(self.checkpoint_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print("✅ Checkpoint initialized successfully")
            return self.checkpoint_file

        except Exception as e:
            print(f"❌ Error initializing checkpoint: {e}")
            print(f"   Checkpoint dir: {self.checkpoint_dir}")
            print(f"   Attempted file: {self.checkpoint_file}")
            raise

    def save_result(self, query_id: str, result: list[int] | None, metadata: dict | None = None) -> None:
        """Save a single evaluation result to the checkpoint.

        Stores the result both in memory cache and persistent checkpoint file.
        Updates the checkpoint file with the new result and processing statistics.

        Args:
            query_id (str): Unique identifier for the query being processed.
            result (list[int] | None): List of ranked indices from the evaluation, or None on error.
            metadata (dict | None, optional): Additional metadata to store with
                the result. Defaults to None.

        Raises:
            ValueError: If checkpoint has not been initialized.

        Note:
            Does not raise exceptions for checkpoint file write failures to
            avoid interrupting the evaluation process.
        """
        if self.checkpoint_file is None:
            msg = "Checkpoint not initialized. Call initialize_checkpoint first."
            raise ValueError(msg)

        try:
            # Add to in-memory cache
            self.processed_ids.add(query_id)
            result_entry = {
                "query_id": query_id,
                "result": result if result is not None else [],
                "processed_at": datetime.datetime.now().isoformat(),
                "metadata": metadata or {},
            }
            self.results_cache.append(result_entry)

            # Update checkpoint file
            with open(self.checkpoint_file) as f:
                checkpoint_data = json.load(f)

            checkpoint_data["total_processed"] += 1
            checkpoint_data["processed_ids"].append(query_id)
            checkpoint_data["results"].append(result_entry)
            checkpoint_data["last_updated"] = datetime.datetime.now().isoformat()

            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

        except Exception as e:
            print(f"⚠️ Warning: Could not save checkpoint for {query_id}: {e}")
            # Don't raise - continue processing even if checkpoint fails

    def load_checkpoint(self, checkpoint_path: str) -> dict | None:
        """Load existing checkpoint to resume processing from previous state.

        Restores the manager state from a previously saved checkpoint file,
        including processed IDs and cached results.

        Args:
            checkpoint_path (str): Path to the checkpoint file to load.

        Returns:
            dict | None: Loaded checkpoint data, or None if file not found.

        Note:
            Updates internal state (processed_ids, results_cache) with loaded data.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"⚠️ Checkpoint file not found: {checkpoint_path}")
            return None

        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        self.checkpoint_file = checkpoint_path
        self.processed_ids = set(checkpoint_data["processed_ids"])
        self.results_cache = checkpoint_data["results"]

        print(f"✅ Loaded checkpoint: {checkpoint_path}")
        print(f"📊 Previously processed: {len(self.processed_ids)} items")

        return checkpoint_data

    def is_processed(self, query_id: str) -> bool:
        """Check if a query ID has already been processed.

        Args:
            query_id (str): The query identifier to check.

        Returns:
            bool: True if the query has been processed, False otherwise.
        """
        return query_id in self.processed_ids

    def get_results(self) -> list[dict]:
        """Get all cached evaluation results.

        Returns:
            list[dict]: List of all processed results with metadata.
                Each result contains query_id, result, processed_at timestamp,
                and optional metadata.
        """
        return self.results_cache

    def export_results(self, output_path: str) -> None:
        """Export all cached results to a separate JSON file.

        Creates a summary file containing all processed results along with
        metadata about the checkpoint file and processing statistics.

        Args:
            output_path (str): Path where the results file should be saved.
                Parent directories will be created if they don't exist.

        Note:
            The exported file includes checkpoint_file path, total_processed
            count, and the complete results array.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "checkpoint_file": str(self.checkpoint_file),
                    "total_processed": len(self.processed_ids),
                    "results": self.results_cache,
                },
                f,
                indent=2,
            )

        print(f"💾 Results exported to: {output_path}")
