# EISFitting

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


the command to be run, from the root of the project:
python manage.py eis_main --mode=import_process_output --logdir=OnePercentTraining --input_dir=RealData\EIS --dataset=USER7 --output_dir=OutputData10
this will look in the folder RealData/EIS for the spectra. 
it will process all the data and record it under a "dataset" called USER7. 
It will then output the results in a folder called OutputData10
If this command is run on different data with the same option for --dataset, 
the data will be stored in the same place and outputted all at once. 
So if you want separate outputs, use different dataset options 
(i.e. --dataset=USER7 the first time and --dataset=USER8 the second time)
Furthermore, one can use different versions of the inverse_model. 
The model itself is contained in a folder. 
By default, a model which used only one percent of the data to 
train is provided in OnePercentTraining. 
To use a different model, replace --logdir=OnePercentTraining 
with --logdir=TenPercentTraining for instance.

The output contains the original spectra, as well as the fitted versions, 
but there is the option to represent impedance as a function of either 
1) Frequency (Hz) or (simply add --no-angular-freq as an option)
2) Angular Frequency (rad/s) (simply add --angular-freq as an option)

for instance, to get angular frequency on the previous data:
python manage.py eis_main --mode=import_process_output --logdir=OnePercentTraining --angular-freq --output_dir=OutputData10 --input_dir=RealData\EIS --dataset=USER7


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
install the requirements by running on the command line "pip install <something>" with <something substituted for a requirement.

# Investigation and New Features

We have identified a problem with softmax over NINF. the solution was to use float32.max instead.
There is still a problem with that code base. 
Now trying without infinities.

We still don't know the impact of having synthetic data. but not having it is not preventing overfitting

We still don't know the impact of having regular convolutions. but having does not prevent overfit.