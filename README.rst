Dataset
=======

Datasets are split into ``train``, ``val`` and ``test`` set.
Every line is a self contained JSON string (ndJSON_).

Scene:

.. code-block:: json

    {"scene": {"id": 266, "p": 254, "s": 10238, "e": 10358}}

Track:

.. code-block:: json

    {"track": {"f": 10238, "p": 248, "x": 13.2, "y": 5.85}}

with:

* ``id``: scene id
* ``p``: pedestrian id
* ``s``: start frame id
* ``e``: end frame id
* ``f``: frame id
* ``x``: x-coordinate in meters
* ``y``: y-coordinate in meters

Frame numbers are not recomputed. Rows are resampled to about
2.5 rows per second.


Tools
=====

* number of scenes per dataset: ``python -m trajnettools.dataset <dataset_files>``
* summary plots per dataset: ``python -m trajnettools.summarize <dataset_files>``
* plot trajectories in a scene: ``python -m trajnettools.trajectories <dataset_file>``


.. _ndJSON: http://ndjson.org/
