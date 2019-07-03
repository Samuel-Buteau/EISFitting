# EISFitting

##Recent Changes
- Now, 24 equivalent circuits are implemented. 
- Optional number of ZARC components (1,2,3)
- Optional inductance 
- Optional ZARC inductance
- The warburg element (modelling diffusion resistance) can now be placed either in series with the ZARCs or nested within the resistance branch of the low frequency ZARC. (this is called "warburg_inception".)

## Important Bugs Fixed
- the parameters 'Q warburg' and 'Q inductance' used to be outputted as 1/Q instead. This is now fixed. 
- Issues with model configurations and pretrained model not being updated have been solved, with internal testing added before release.

## The Paper
Analysis of Thousands of Electrochemical Impedance Spectra of Lithium-Ion Cells through a Machine Learning Inverse Model
by Sam Buteau and J. R. Dahn

doi: 10.1149/2.1051908jes
J. Electrochem. Soc. 2019 volume 166, issue 8, A1611-A1622

This repository was released together with a paper, which explains the approach taken and some of the terminology.

# Usage

The current version of the software comes with a pretrained model, 
and can be run on a given Folder. By running the code on a given Folder, 
1) all the recognised files will be imported into the internal database. 
(if a user wants to use a different format, 
only the import function needs to be rewritten to fill the database with the right data format)
2) the inverse model will be run all recognised spectra.
3) the finetuning will be run on the resulting parameters. 
4) the fits will be outputted as a png (plot) and as a CSV (for publication quality plotting), 
    and the circuit parameters extracted by these processes will be outputted to a CSV file 
    where each row represents a different impedance spectrum. Three sets of parameters are given.
    1) The parameters produced by the inverse model.
    2) The parameters produced by finetuning.
    3) The difference between the two. (this is a proxy for the uncertainty in those fitting parameters.)

## Typical usage
the command to be run, from the root of the project:
python manage.py eis_main --mode=import_process_output --logdir=OnePercentTraining --input_dir=RealData\EIS --dataset=USER7 --output_dir=OutputData10

this will look in the folder RealData/EIS for the spectra. 
it will process all the data and record it under a "dataset" called USER7. 
It will then output the results in a folder called OutputData10


## Separating the outputs and organizing data into datasets
If this command is run on different data with the same option for --dataset, 
the data will be stored in the same place and outputted all at once. 

If you want separate outputs, use different dataset options 
(i.e. --dataset=USER7 the first time and --dataset=USER8 the second time)


## Specifying a different inverse model
Furthermore, one can use different versions of the inverse_model. 
The model itself is contained in a folder. 
By default, a model which used only one percent of the data to 
train is provided in OnePercentTraining. 
To use a different model, replace --logdir=OnePercentTraining 
with --logdir=TenPercentTraining for instance.

## Angular frequency or plain frequency
The output contains the original spectra, as well as the fitted versions, 
but there is the option to represent impedance as a function of either 
1) Frequency (Hz) or (simply add --no-angular-freq as an option)
2) Angular Frequency (rad/s) (simply add --angular-freq as an option)

for instance, to get angular frequency on the previous data:
python manage.py eis_main --mode=import_process_output --logdir=OnePercentTraining --angular-freq --output_dir=OutputData10 --input_dir=RealData\EIS --dataset=USER7


## Specifying the equivalent circuit

By default, the equivalent circuit used has 3 ZARC components, and it does not have an inductance nor does it have a zarc inductance.
If a different number of ZARC components are desired (e.g. 2), simply add e.g. --num_zarcs=2. this would give 
python manage.py eis_main --mode=import_process_output --logdir=OnePercentTraining --input_dir=RealData\EIS --dataset=USER7 --output_dir=OutputData10 --num_zarcs=2

- If an inductance is desired, simply pass --inductance
- If a zarc inductance is desired, simply pass --zarc-inductance
These can both be passed if both components are desired.

- If the warburg element (modelling diffusion) should be nested within the low frequency ZARC (the ZARC element becoming a parallel circuit of a CPE in parallel with a serial circuit of a resistor with the warburg element in series), simply pass --warburg_inception
For instance, if you would like a zarc_inductance, 1 electrochemical ZARC, with the warburg nested inside, you would use:
python manage.py eis_main --mode=import_process_output --logdir=OnePercentTraining --input_dir=RealData\EIS --dataset=USER7 --output_dir=OutputData10 --num_zarcs=1 --zarc_inductance --warburg_inception

# Locations
## Folders
- OnePercentTraining contains a pretrained model. 
- EIS contains code
    - EIS/models.py defines how the dataset is stored.
    - EIS/management/commands/eis_main.py is where the code functionality is defined
     
## Files
- .gitattributes can be ignored
- .gitignore can be ignored
- db.sqlite3 holds the data provided by the user. The program handles interactions with this file. Please leave alone.
- LICENSE contains the license
- manage.py is the entry point for the program.
- requirements.txt contains the requirements. (excluding python 3.6)



# Requirements
see requirements.txt
install the requirements by running on the command line "pip install *something*" with *something* substituted for a requirement.

# Notes to self

We must make sure that the text user interface works with the upgraded model.
Then we must make a basic graphical user interface for plain data to allow visualization and correction of bad data.
