This project contains the code used to generate the explanatory math videos found on [3Blue1Brown](https://www.3blue1brown.com/).

This almost entirely consists of scenes generated using the library [Manim](https://github.com/3b1b/manim).  See also the community maintained version at [ManimCommunity](https://github.com/ManimCommunity/manim/).

Note, while the library Manim itself is open source and under the MIT license, the contents of this project are intended only to be used for 3Blue1Brown videos themselves.

Copyright © 2022 3Blue1Brown

## Installation
1. Install `manimgl` [from source](https://github.com/3b1b/manim?tab=readme-ov-file#directly-windows) since the latest version published is not up to date.
2. Ensure Latex is installed, for example on Ubuntu
   ```sh
   sudo apt install texlive
   sudo apt install texlive-latex-extra
   sudo apt install texlive-fonts-extra
   sudo apt install texlive-science
   ```
3. Clone this repository to run the video samples
   ```sh
   git clone git@github.com:3b1b/videos.git
   cd videos
   ```

### Running a video

1. Ensure Latex is installed, for example on Ubuntu
   ```sh
   sudo apt install texlive-latex-extra
   sudo apt install texlive-fonts-extra
   ```

2. Setup includes (if you don't want to modify your path)
    ```sh
    cd _2023/optics_puzzles
    vi e_field.py
    ```
    And add the following to the top
    ```python
    import sys
    sys.path.append(".")
    sys.path.append("..")
    sys.path.append("../..")
    ```

3. Run an example
   ```sh
   manimgl e_field.py
   ```
   Or
   ```sh
   manimgl e_field.py WavesIn3D
   ```
