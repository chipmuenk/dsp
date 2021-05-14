# Unterlagen zu den Kursen "DSV" und "DSV auf FPGAs" 
[Course material accompanying the courses "DSP" and "DSP on FPGAs" (Digital signal processing on FPGAs)]

**ATTENTION:** If you have cloned / forked this repo, it has been renamed from `dsp_fpga` -> `dsp` and the default branch now is called `main` (2020-Sep-29).

Hier finden Sie die folgenden Materialien:
* Jupyter Notebooks zu beiden Lehrveranstaltungen und zum YouTube Channel [https://www.youtube.com/c/ChristianMunker](https://www.youtube.com/c/ChristianMunker)
* Ein [Skript](docs/DSV_FPGA_Muenker_Skript.pdf) mit vielen Übungsaufgaben und etwas Theorie zu den Kursen "DSV" und "DSV auf FPGAs"
* Eine [Anleitung](docs/2020-DSP_Notebooks) zum Umgang mit Notebooks

Nutzen Sie auch das interaktive Python Tool [pyFDA](https://github.com/chipmuenk/pyfda) für Filterentwurf und -analyse und zur Simulation von zeitdiskreten Systemen!

### Jupyter Notebooks

* Kurzanleitung: https://codingthesmartway.com/getting-started-with-jupyter-notebook-for-python/ mit Video https://youtu.be/CwFq3YDU6_Y
* Jupyter Notebooks: Ein weiteres sehr gutes Video Tutorial zu Jupyter Notebooks finden Sie unter [https://www.youtube.com/watch?v=HW29067qVWk]

**[00. INTRO:](notebooks/00_Intro/_INTRO-Index.ipynb)** Eine kurze interaktive Einführung in Notebooks, Numpy, Scipy, Matplotlib

**[00. LAB :](notebooks/00_LAB/_index.ipynb)** Praktikumsversuche (als Jupyter Notebooks)

**[01. LTI :](notebooks/01_LTI/_LTI-Index.ipynb)** Linear Time-Invariant (**LTI**) Systeme im Zeitbereich

**[02. LTF :](notebooks/02_LTF/_LTF-Index.ipynb)** **LT**I Systeme im **F**requenzbereich

**[03. DFT :](notebooks/03_DFT/_DFT-Index.ipynb)** Discrete Fourier Transformation (**DFT**) und FFT

**[04. WIN :](notebooks/04_WIN/_index.ipynb)** Fensterung periodischer und stationärer Signale

**[05. SPS :](notebooks/05_SPS/_index.ipynb)** **SP**ektral**S**chätzung

**[06. FIL :](notebooks/06_FIL/_index.ipynb)** Digitale **FIL**ter und Filterentwurf

**[07. FIX :](notebooks/07_FIX/_index.ipynb)** **FIX**point Systeme im Zeitbereich: Quantisierung und Wortlängeneffekte

**[08. NOI :](notebooks/08_NOI/_index.ipynb)** Fixpoint Systeme im Frequenzbereich: Quantization **NOI**se

**[09. SMP :](notebooks/09_SMP/_index.ipynb)** **S**a**MP**ling, Analog-Digital Conversion and Downsampling

**[10. INP :](notebooks/10_INP/_index.ipynb)** Upsampling, **IN**ter**P**olation und Digital-Analog conversion

**[11. SRC :](notebooks/11_SRC/plots)** **S**ample **R**ate **C**onversion

## Jupyter Notebook Server in der Cloud
Am einfachsten können Sie mit Jupyter Notebooks interaktiv auf einem Remote Server arbeiten, Sie müssen dann nichts auf Ihrem eigenen Rechner installieren und können einfach im Browser arbeiten, müssen aber natürlich online sein. In der Vergangenheit haben die beiden folgenden Server am besten funktioniert.

### Gesis Notebook Server

Erstellen Sie einen kostenlosen Account unter https://notebooks.gesis.org oder unter http://notebooks.gesis.org/services/binder/v2/gh/chipmuenk/dsp/main. Bei letzterem wird dieses Github Repo fertig für Sie eingerichtet!

### Microsoft Codespaces

Anscheinend können Notebooks auch können portiert werden zu https://github.com/features/codespaces .

### Einrichten Ihres Servers
Installieren Sie im Terminal des Servers die folgenden Python Module nach:

    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install nbdime

Dann clonen Sie dieses Repo aus dem Terminal mit `git clone https://github.com/chipmuenk/dsp`.

Wenn Sie lieber ein Jupyterlab Interface haben möchten, ersetzen Sie in der Adresszeile `tree` durch `lab`.

## Lokal arbeiten
Wenn Sie Python auf Ihrem Rechner installiert haben, können Sie auch offline arbeiten und haben eine bessere Performance. Die Notebooks (und die Libraries) clonen Sie auf Ihren Rechner aus dem (lokalen) Terminal mit 

    git clone https://github.com/chipmuenk/dsp
    
Oder nutzen Sie die graphische Oberfläche mit `git gui` -> `Clone Repository`
  
Dazu muss ein git Client von der git homepage (http://git-scm.com/) auf Ihrem Rechner installiert sein.

Notfalls können Sie die Files auch gezippt herunterladen von  https://github.com/chipmuenk/dsp, können dann aber keine Updates holen.


### git
Es schadet nicht, ein paar git Kommandos zu beherrschen, z.B. mit Hilfe von

* [git - Der einfache Einstieg](http://rogerdudler.github.io/git-guide/index.de.html) von Roger Dudler gibt den kürzest möglichen Einstieg in die Git Bash (= Kon-
sole) - mit Cheat-Sheet! - in vielen Sprachen
* [An Illustrated Guide to Git on Windows](http://nathanj.github.io/gitguide/tour.html) (2009) gibt einen ähnlich kompakten Einstieg in die Arbeit mit dem graphischen Frontend Git GUI
* [Pro Git Book](http://git-scm.com/book/de/v2), das "offizielle" Git Buch von Scott Chacon und Ben Straub gibt es hier in ziemlich vielen Sprachen
* [Learn Git Branching](https://learngitbranching.js.org?locale=de_DE) ist eine „gamifizierte“ Variante mit Schwerpunkt Branching und Merging (auch auf Deutsch)
