# Unterlagen zum Kurs "DSV auf FPGAs" 
[Course material accompanying the lecture "DSP on FPGAs" (Digital signal processing on FPGAs)]

Hier finden Sie die folgenden Materialien:

* Folien zur Vorlesung und den Screencasts auf dem YouTube Channel [https://www.youtube.com/channel/UCsnhY3xQqxw8tjpsAH_vBeA](https://www.youtube.com/channel/UCsnhY3xQqxw8tjpsAH_vBeA)
* Übungsaufgaben und (etwas) Theorie
* Python und Matlab Files

### Python und Matlab Files
Here, you find python snippets to demonstrate various aspects of digital signal processing. Don't expect too much "FPGA" in the code snippets, the focus is more on general digital signal processing with some fixpoint arithmetics. The chapters and code snippets are labeled as following:

**[1. LTI :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/1_LTI)** Linear Time-Invariant systems in the time domain

**[2. LTF :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/2_LTF)** LTI systems in the Frequency domain

**[3. DFT :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/3_DFT)** Discrete Fourier Transformation and FFT

**[4. FIL :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/4_FIL)** Digital FILters and filter design

**[5. FIX :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/5_FIX)** FIXpoint systems in the time domain: Quantization and word length effects 

**[6. NOI :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/6_NOI)** Fixpoint systems in the frequency domain: Quantization NOIse

**[7. SMP :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/7_SMP)** SaMPling, Analog-Digital Conversion and Downsampling

**[8. INP :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/8_INP)** Upsampling, INterPolation and Digital-Analog conversion

**[9. SRC :](https://github.com/chipmuenk/dsp_fpga/tree/master/code/9_SRC)** Sample Rate Conversion

## Getting started
You can either download the zip-File containing the most up-to-date version of the files - then you can stop reading here.

Or - better - clone this repository to your local computer. For doing this, you need to 

* **Create a GitHub account** (this file hosting platform) under [https://github.com/]
* **Install a git client** from the git homepage [http://git-scm.com/], providing a console (`git bash`) and a graphical (`Git GUI`) frontend to git.
* **Get to know a few basic commands**, a good introductory & interactive course is e.g. [Ry’s Git Tutorial](http://rypress.com/tutorials/git/index).
* **Clone the repository:**
  Start the `git bash` from the local directory where you want to copy the data to and enter:

  ```
  git clone https://github.com/chipmuenk/dsp_fpga
  ```
  
  Or start `Git GUI` -> Clone Repository.  
