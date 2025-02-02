# Task B: Comparative Spectrogram Analysis of Songs

This folder contains all the resources needed to **analyze and compare the spectrograms** of songs from four different genres. You will select your own songs, generate their spectrograms, and then provide a detailed comparative analysis.

## Contents

- **songs/**  
  The directory where you will place your **four audio files** (one for each genre).

- **spectrograms/**  
  This is where the generated spectrogram images will be stored after running the spectrogram-generation script.

- **gen_spec.py**  
  The main script to generate spectrograms for all audio files found in the `songs/` folder.  
  - In this script, you can update the input and output paths as needed:
    ```python
    if __name__ == "__main__":
        generate_spectrograms(
            '/data/b22cs089/speechPA1/taskb/songs',
            '/data/b22cs089/speechPA1/taskb/spectograms'
        )
    ```
    Change these paths to match your local directory structure if necessary.

- **hamming.py**, **hann.py**, **rectangular.py**  
  Scripts demonstrating how to apply different windowing functions (Hamming, Hann, Rectangular) for audio processing.  
  - These may be helpful if you want to experiment with how windowing affects spectrogram generation.

- **README.md**  
  The file you’re reading right now.

## How to Use

1. **Add Your Audio Files**  
   - Place **four songs** (one from each of your chosen genres) inside the `songs/` folder.
   - Ensure they are in a compatible format (e.g., WAV, MP3).

2. **Adjust Paths (If Needed)**  
   - Open `gen_spec.py` and update the input folder (`songs`) and the output folder (`spectrograms`) paths to reflect your environment.

3. **Generate Spectrograms**  
   - From the command line, run:
     ```bash
     python gen_spec.py
     ```
   - This script will read each song from the `songs/` folder, create a spectrogram image, and save it into the `spectrograms/` folder.

4. **Compare Spectrograms**  
   - Open the generated images in `spectrograms/`.  
   - Observe how the frequency components differ across time for each genre.

5. **Analysis**  
   - The analysis of my generated spectrograms can be found [here](https://chatgpt.com).


