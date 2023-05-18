# Large Volume Visualization
This is an implementation of a CPU large volume visualization pipeline, which consists of a data structure for large volumetric data and an algorithm that visualizes such data. The hierarchical data structure accelerates sampling and allows the reduction of the total amount of data that needs to be loaded into physical memory during visualization. Visualization of processed data is achieved by the ray casting method with existing optimization techniques, such as empty space skipping and early ray termination. The project also contains tools for working with RAW volume files.
The pipeline is described and evaluated in [1].

## Compile
The project depends on ``GCC`` with C++20 implemented. Dependencies are resolved via the the ``conan`` package manager. The compilation utilizes the ``cmake`` tool. The compilation is fairly straightforward:

    conan install --settings build_type=Release --output-folder build --build missing .
    cmake --preset conan-release
    cmake --build --preset conan-release

## Usage
The tools can convert and visualize large volumetric datasets.
To convert raw volumetric data into the format used for visualization, use processing tool like this:

    ./tools/process path/to/input.raw <width> <height> <depth> <bytes-per-voxel> path/to/output.blocks path/to/output.tree

The ``<width>``,  ``<height>``, ``<depth>`` and ``<bytes-per-voxel>`` parameters describes the volume data.
The tool creates two files, ``path/to/output.blocks`` and ``path/to/output.tree``, that represents the data structures used for visualization.

The visualization tool can visualize processed data structures with the corresponding visualization tool.
To vizualize processed volume, use visualization tool like this:

    ./tools/vizualize path/to/output.blocks path/to/output.tree <width> <height> <depth> <bytes-per-voxel> path/to/trasfer/function.tf

The tool maps two files, ``path/to/output.blocks`` and ``path/to/output.tree``, into memory.
Because the files are more of a data container than formats, the tool needs to know the description of the data via ``<width>``,  ``<height>``, ``<depth>`` and ``<bytes-per-voxel>`` parameters.
The ``path/to/trasfer/function.tf`` is a file that describes transfer function.

The tool consists of a viewport and a small debugging user interface.
The movement through the scene can be achieved with ``WASD`` controls.
The rotation of the camera can be achieved by holding the right mouse button and moving the mouse.
The UI allows for custom volume and camera transformations also.

Besides the transformations, the UI can be used to configure the renderer.
The integration step value ``Step`` and early ray termination value ``Ray termination threshold`` can be configured.
The UI allows for the selection of the rendering methods ``Scalar Tree``, ``Vector Tree`` and ``Packlet Tree`` that differ by the way of the casting of rays and vectorization.
Other than that, there are some more experimental renderers.
The parameter ``Quality`` controls the data hierarchy selection algorithm.

The transfer function file is in format:

    <number-of-colors>
    <key> <red> <green> <blue>
    ...
    <number-of-values>
    <key> <alpha>
    ...

Where the ``<number-of-colors>`` and ``<number-of-values>`` define the number of key points.
The key points ``<key>`` are values from interval <0, 1>, where 0 corresponds to the lowest voxel value and 1 corresponds to the highest voxel value.
This way, the transfer function works both for 8-bit and 16-bit data.
Color points ``<red>``, ``<green>`` and ``<blue>`` are values fron interval <0, 1> that together represents one color from rgb color space.
Alpha values ``<alpha>`` are positive values that define coefficient of the opacity of the sample, meaning higher values is more opaque.

## License
See [LICENSE](LICENSE) for license and copyright information.

## References
[1] Dlabaja, D.: Zobrazení rozsáhlých volumetrických dat na CPU. Brno, 2021. Diplomová práce. Vysoké učení technické v Brně, Fakulta informačních technologií. Vedoucí práce
Ing. Michal Španěl, Ph.D.
