def shard_streaming_dataset(ds, rank, world_size):
    return ds.filter(
        lambda _, idx: idx % world_size == rank,
        with_indices=True,
    )
