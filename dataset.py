from datasets import load_dataset


class Dataset:
    """
    Dataset class to load the chunked dataset from Hugging Face Datasets
    """
    def __init__(self, dataset_name: str, split: str) -> None:
        self.dataset_name = dataset_name
        self.split = split

    def get_dataset(self) -> list:
        dataset = load_dataset(self.dataset_name, split=self.split)
        dataset = dataset.map(lambda x: {
            "id": f'{x["id"]}-{x["chunk-id"]}',
            "text": x["chunk"],
            "metadata": {
                "title": x["title"],
                "url": x["source"],
                "primary_category": x["primary_category"],
                "published": x["published"],
                "updated": x["updated"],
                "text": x["chunk"],
            }
        })
        # drop uneeded columns
        dataset = dataset.remove_columns([
            "title", "summary", "source",
            "authors", "categories", "comment",
            "journal_ref", "primary_category",
            "published", "updated", "references",
            "doi", "chunk-id",
            "chunk"
        ])
        return dataset
