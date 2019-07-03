del db.sqlite3
del EIS\migrations\*
copy NUL EIS\migrations\__init__.py
python manage.py makemigrations
python manage.py migrate
python manage.py edit_database --mode=add_default_inverse_models