# Unterlagen zu den Kursen "DSV" und "DSV auf FPGAs" 
[Course material accompanying the lectures "DSP" and "DSP on FPGAs" (Digital signal processing on FPGAs)]

Hier finden Sie die folgenden Materialien:

* Folien zur Vorlesung und den Screencasts auf dem YouTube Channel [https://www.youtube.com/c/ChristianMunker](https://www.youtube.com/c/ChristianMunker)
* Viele Übungsaufgaben und (etwas) Theorie zur Vorlesung "DSV auf FPGAs"

* Eine Anleitung zum Umgang mit Notebooks

### Unterlagen zum Umgang mit Jupyter Notebooks ###

* Kurzanleitung: https://codingthesmartway.com/getting-started-with-jupyter-notebook-for-python/ mit Video https://youtu.be/CwFq3YDU6_Y
* Jupyter Notebooks: Ein sehr gutes Video Tutorial zu Jupyter Notebooks finden Sie unter [https://www.youtube.com/watch?v=HW29067qVWk]

### Jupyter Notebooks zur Vorlesung

**[01. LTI :](notebooks/01_LTI)** Linear Time-Invariant (**LTI**) Systeme im Zeitbereich

**[02. LTF :](notebooks/02_LTF)** **LT**I Systeme im **F**requenzbereich

**[03. DFT :](notebooks/03_DFT)** Discrete Fourier Transformation (**DFT**) und FFT

**[04. WIN :](notebooks/04_WIN)** Fensterung periodischer und stationärer Signale

**[06. FIL :](notebooks/06_FIL)** Digitale **FIL**ter und Filterentwurf

**[07. FIX :](notebooks/07_FIX)** **FIX**point Systeme im Zeitbereich: Quantisierung und Wortlängeneffekte

**[08. NOI :](notebooks/08_NOI)** Fixpoint Systeme im Frequenzbereich: Quantization **NOI**se

**[09. SMP :](notebooks/09_SMP)** **S**a**MP**ling, Analog-Digital Conversion and Downsampling

**[10. INP :](notebooks/10_INP)** Upsampling, **IN**ter**P**olation und Digital-Analog conversion

**[11. SRC :](notebooks/11_SRC/plots)** **S**ample **R**ate **C**onversion

## Jupyter Notebook Server in der Cloud
Am einfachsten können Sie mit Jupyter Notebooks interaktiv auf einem Remote Server arbeiten, Sie müssen dann nichts auf Ihrem Rechner installieren und können einfach im Browser arbeiten, müssen aber natürlich online sein. In der Vergangenheit haben die beiden folgenden Server am besten funktioniert.

### Gesis Notebook Server

Erstellen Sie einen kostenlosen Account unter https://notebooks.gesis.org 

### Microsoft Azure Notebooks

Melden Sie sich mit Ihren Microsoft Konto an bei https://notebooks.azure.com (manchmal überlastet).

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

Es schadet auch nicht, ein paar git Kommandos zu beherrschen, z.B. mit Hilfe von [Ry’s Git Tutorial](http://rypress.com/tutorials/git/index).



```python

```
