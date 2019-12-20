# Q2Q: Improving Relevance of Responses by Converting Queries to Questions

Code can be found at: [https://github.com/sominwadhwa/cs585-q2q](https://github.com/sominwadhwa/cs585-q2q)

This repository contains all the code written for the final project of `COMPSCI 585: Intro to Natural Language Processing (Mohit Iyyer)` in the Fall 2019.

**Authors**: Thivakkar Mahendran, Vincent Pun, Kathryn Ricci, Apurva Bhandari, Somin Wadhwa

## Pre-requisites

The following are a couple of instructions that must be gone through in order to execute different (or all) sections of this project.

1. Clone the project, replacing ``cs585`` with the name of the directory you are creating:

        $ git clone https://github.com/sominwadhwa/cs585-q2q.git cs585
        $ cd drugADR

2. Make sure you have ``python 3.6.x`` running on your local system. If you do, skip this step. In case you don't, head
head [here](https://www.python.org/downloads/).

3. ``virtualenv`` is a tool used for creating isolated 'virtual' python environments. It is advisable to create one here as well (to avoid installing the pre-requisites into the system-root). Do the following within the project directory:

        $ [sudo] pip install virtualenv
        $ virtualenv --system-site-packages cs585
        $ source cs585/bin/activate

To deactivate later, once you're done with the project, just type ``deactivate``.

4. Install the pre-requisites from ``requirements.txt`` & run ``test/init.py`` to check if all the required packages were correctly installed:

        $ pip install -r requirements.txt
        $ python test/init.py

You should see a message - ``Imports successful. Good to go!``

## Directory Structure

#### Top-Level Structure:

    .
    .
    ├── data                     # Data used and/or generated
    │   ├── dev_q2d.txt
    │   ├── train_q2d.txt
    │   ├── google_queries.txt
    │   ├── squad_queries.txt
    │   ├── hotpot_queries.txt
    ├── preprocessing-src                     # Source Files
    │   ├── __init__.py
    │   ├── get_context.py
    │   ├── preprocess.py
    ├── src                    # Main source files for training
    │   ├── __init__.py
    │   ├── encoderdecoder.py
    │   ├── failure-analysis-ids.py
    │   ├── keras-nmt-baseline-wo-attn.py
    │   ├── train-eval-q2q-w-attn.py     
    ├── attn_ex (<num>).png 
    ├── attn_ex_sp<num>.png
    ├── loss_curve.png                  
    ├── LICENSE
    └── README.md
    .
    .



