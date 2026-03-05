def shard_streaming_dataset(ds, rank: int, world_size: int):
    """Keep only the rows that belong to this rank (round-robin sharding)."""
    if world_size <= 1:
        return ds
    return ds.filter(
        lambda _, idx: idx % world_size == rank,
        with_indices=True,
    )
