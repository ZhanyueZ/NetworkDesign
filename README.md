# Network Designer: a pratical implementation of graph theory
This program is a collabrative project of ECSE 422 Fault Tolerant Computing

## How to use:
#### 1. Modify the `path` attribute in the NetworkDesigner.py (individual tester files under folder 'tester');
#### 2. Install dependencies as instructed in requirements.txt. You may do this by running the command below;
```
pip install -r requirements.txt
```
#### 3. Input the cost limit. Beware of your cost estimation before trying;
#### 4. The runtime comaprison should be printed and network designs for each version including maximum reliability should be plotted.

NOTE THAT trying a tester with large number of cities (8_1.txt for instance) or specifying a huge cost can lead the simple algorithm to run 
for an unacceptably long period, indicated by a progress bar on terminal. You may as well comment out step 4 in main() under such scenario.
