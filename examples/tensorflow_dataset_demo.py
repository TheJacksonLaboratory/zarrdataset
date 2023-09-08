import numpy as np
import zarrdataset as zds
import time

try:
    import tensorflow as tf

    if __name__ == "__main__":
        print("Zarr-based data loader demo")
        # These are images from the Image Data Resource (IDR)
        # https://idr.openmicroscopy.org/ that are publicly available and were 
        # converted to the OME-NGFF (Zarr) format by the OME group. More
        # examples can be found at Public OME-Zarr data (Nov. 2020)
        # https://www.openmicroscopy.org/2020/11/04/zarr-data.html
        filenames = ["https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0073A/9798462.zarr"]

        patch_size = (512, 512)
        batch_size = 4
        num_workers = 0

        patch_sampler = zds.GridPatchSampler(patch_size)

        my_dataset = zds.ZarrDataset(filenames, transform=transform_fn,
                                     data_group="2",
                                     source_axes="TCZYX",
                                     roi="(0,0,0,0,0):(1,-1,1,-1,-1)",
                                     axes="YXC",
                                     patch_sampler=patch_sampler)

        my_dataloader = tf.data.Dataset.from_generator(
            my_dataset.__iter__,
            output_signature=(tf.TensorSpec(shape=(None, None, 3),
                                            dtype=tf.float32),
                              tf.TensorSpec(shape=(),
                                            dtype=tf.int64)))

        batched_dataset = my_dataloader.batch(batch_size)

        etimes = 0
        etime = time.perf_counter()
        for i, (x, t) in enumerate(batched_dataset):
            etime = time.perf_counter() - etime
            etimes += etime
            print("Sample %i" % i, x.shape, x.dtype, t.shape, t.dtype, etime,
                  etimes / (i + 1))
            etime = time.perf_counter()

except ModuleNotFoundError:
    import logging
    logging.warning("Tensorflow is not installed and this demo requires it to "
                    "work, please use 'general_dataset_demo.py instead'")
