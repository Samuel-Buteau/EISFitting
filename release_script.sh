rm db.sqlite3
rm -f EIS/migrations/*
touch EIS/migrations/__init__.py
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py edit_database --mode=add_default_inverse_models
python3 manage.py edit_database --mode=add_default_file_formats