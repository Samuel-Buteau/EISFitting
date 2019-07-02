rm db.sqlite3
python manage.py migrate
python manage.py edit_database --mode=add_default_inverse_models