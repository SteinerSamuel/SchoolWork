from typing import List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import json
import datetime

app = Flask(__name__)
CORS(app)
"""
This is a restful API which will be consumed by the frontend in production this should be protected by using
Keycloak authorization client or another authorization method but because of time it is not implemented.
On top of this config for the MySql DB should be defined in envirioment variables and not in plain text as part of
the file this can be done within the docker-compose file again not implemented due to time.

Each function below corosponds to each api end point also down below
above we intitate CORS which stands for Cross Origin Resource Sharing this allows us to consume th api from a different URL
"""
def titlesearch(Title) -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
        bi.title,
        a.f_name ,
        a.l_name,
   	    SUM(1 - bi2.Checked_out) AS available,
   	    count(*) AS total,
        bi.ISBN,
        bi.desc AS Description,
        bi.number_of_pages,
        bi.cover_url
    FROM
        default_schema.book_info bi
    INNER JOIN default_schema.book_Inventory bi2 ON
        bi.ISBN = bi2.book_info_ISBN
    INNER JOIN default_schema.author a ON
        bi.author_author_id = a.author_id
    WHERE
	    bi.title LIKE '%"""+ Title  +"""%'
    GROUP BY
	    bi2.book_info_ISBN
    """
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
                "Title": Title,
                "Author" : f_name + " " + l_name ,
                "Available": int(available),
                "Total": total_Count,
                "Pages": pages,
                "ISBN": ISBN,
                "Description": Description,
                "CoverURL" : url
                }  
                for (Title, f_name, l_name, available, total_Count, ISBN, Description, pages, url) in cursor]
    cursor.close()
    connection.close()
    return results

def authorsearch(author) -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
        bi.title,
        a.f_name ,
        a.l_name,
   	    SUM(1 - bi2.Checked_out) AS available,
   	    count(*) AS total,
        bi.ISBN,
        bi.desc AS Description,
        bi.number_of_pages,
        bi.cover_url
    FROM
        default_schema.book_info bi
    INNER JOIN default_schema.book_Inventory bi2 ON
        bi.ISBN = bi2.book_info_ISBN
    INNER JOIN default_schema.author a ON
        bi.author_author_id = a.author_id
    WHERE
	    CONCAT(a.f_name, ' ' ,a.l_name) LIKE '%"""+ author +"""%'
    GROUP BY
	    bi2.book_info_ISBN
    """
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
                "Title": Title,
                "Author" : f_name + " " + l_name ,
                "Available": int(available),
                "Total": total_Count,
                "Pages": pages,
                "ISBN": ISBN,
                "Description": Description,
                "CoverURL" : url
                }  
                for (Title, f_name, l_name, available, total_Count, ISBN, Description, pages, url) in cursor]
    cursor.close()
    connection.close()
    return results

def isbnsearch(isbn) -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
        bi.title,
        a.f_name ,
        a.l_name,
   	    SUM(1 - bi2.Checked_out) AS available,
   	    count(*) AS total,
        bi.ISBN,
        bi.desc AS Description,
        bi.number_of_pages,
        bi.cover_url
    FROM
        default_schema.book_info bi
    INNER JOIN default_schema.book_Inventory bi2 ON
        bi.ISBN = bi2.book_info_ISBN
    INNER JOIN default_schema.author a ON
        bi.author_author_id = a.author_id
    WHERE
	     bi.ISBN LIKE '%"""+ isbn +"""%'
    GROUP BY
	    bi2.book_info_ISBN
    """
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
                "Title": Title,
                "Author" : f_name + " " + l_name ,
                "Available": int(available),
                "Total": total_Count,
                "Pages": pages,
                "ISBN": ISBN,
                "Description": Description,
                "CoverURL" : url
                }  
                for (Title, f_name, l_name, available, total_Count, ISBN, Description, pages, url) in cursor]
    cursor.close()
    connection.close()
    return results

def get_book_info(isbn) -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
        bi.ISBN,
        bi.yr_publication,
        bi.publisher_publisher_name,
        bi.title,
        bi.desc,
        bi.number_of_pages,
        bi.cover_url,
        a.f_name,
        a.l_name
    FROM
        default_schema.author a
    INNER JOIN default_schema.book_info bi ON
        a.author_id = bi.author_author_id
    WHERE
        bi.ISBN = """ + ' "' + isbn + '"'
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
                "ISBN": ISBN,
                "PublicationYear" : yr_publicaiton,
                "Publisher": publisher_name,
                "Title": title,
                "Pages":pages,
                "Description": desc,
                "CoverURL" : cover_url,
                "Author": f_name + " " + l_name
                }  for (ISBN, yr_publicaiton, publisher_name, title, desc, pages, cover_url, f_name, l_name) in cursor]
    cursor.close()
    connection.close()
    return results

def get_available_books(isbn) -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
        bi.Inventory_id,
        bi.Checked_out,
        bi2.title,
        CONCAT(a.f_name, ' ', a.l_name)
    FROM
        default_schema.author a
    INNER JOIN default_schema.book_info bi2 ON
        a.author_id = bi2.author_author_id
    JOIN default_schema.book_Inventory bi ON
        bi2.ISBN = bi.book_info_ISBN
    WHERE
        bi.Checked_out = 0
            AND bi2.ISBN = """ + ' "' + isbn + '"'
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
                "Inventory_Id": Inventory_number,
                "CheckedOut": checked,
                "Title": title,
                "Author": author
                }  for (Inventory_number, checked, title, author) in cursor]
    cursor.close()
    connection.close()
    return results

def get_rented_books(user) -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
        bi.Inventory_id,
        DATE_FORMAT(bi.checked_out_date,"%Y-%m-%d"),
        DATE_FORMAT(DATE_ADD(bi.checked_out_date, INTERVAL 20 DAY),"%Y-%m-%d"),
        bi2.title,
        CONCAT(a.f_name, ' ', a.l_name),
        bi2.ISBN
        
    FROM
        default_schema.author a
    INNER JOIN default_schema.book_info bi2 ON
        a.author_id = bi2.author_author_id
    JOIN default_schema.book_Inventory bi ON
        bi2.ISBN = bi.book_info_ISBN
    WHERE
        bi.Checked_out_by = """ + ' "' + user + '"'
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
                "Inventory_Id": Inventory_number,
                "Title": title,
                "OutDate": checked_out,
                "DueDate": due_date,
                "Author": author,
                "Available": 0,
                "Total": 0, # Dummy values needed to display in ui
                "ISBN": ISBN
                }  for (Inventory_number, checked_out, due_date, title, author, ISBN)  in cursor]
    cursor.close()
    connection.close()
    return results

def get_all_authors() -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
    *
    FROM
    default_schema.author a
    """
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
        "Author_Id": Author_Id,
        "Name": f_name + ' ' + l_name
        } for (Author_Id, f_name, l_name, bio) in cursor]
    cursor.close()
    connection.close()
    return results

def get_all_publ() -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
    *
    FROM
    default_schema.publisher
    """
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
        "Publisher_name": name
        } for (name, country, city, street, zip_c, state) in cursor]
    cursor.close()
    connection.close()
    return results

def get_all_states() -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
    *
    FROM
    default_schema.state
    """
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
        "State_Abrv": Abrv,
        "Name": name
        } for (Abrv, name) in cursor]
    cursor.close()
    connection.close()
    return results

def get_all_countries() -> List[Dict]:
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    SELECT
    *
    FROM
    default_schema.country
    """
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    results = [{
        "Country_Abrv": Abrv,
        "Name": name
        } for (name, Abrv) in cursor]
    cursor.close()
    connection.close()
    return results

def rent_book(inv_id, user):
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    UPDATE
        book_Inventory
    SET
        Checked_out = 1, Checked_out_by = """ + '"' + user + '"'+ ', checked_out_date= "' + datetime.date.today().strftime("%Y-%m-%d") +'"  ' +"""
    WHERE
        Inventory_id = """ +'"' + inv_id + '"'
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()
    results = [{"records":cursor.rowcount}]
    cursor.close()
    connection.close()
    return results

def return_book(inv_id):
    config = {
        'user': 'root',
        'password': 'root',
        'host': 'db',
        'port': '3306',
        'database': 'default_schema'
    }
    query = """
    UPDATE
        book_Inventory
    SET
        Checked_out = 0, Checked_out_by = NULL , checked_out_date= NULL
    WHERE
        Inventory_id = """ +'"' + inv_id + '"'
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()
    results = [{"records":cursor.rowcount}]
    cursor.close()
    connection.close()
    return results

@app.route('/author/')
def author() -> str:
    author = request.args.get('name', default = '%', type=str)
    response = jsonify({'inventory': authorsearch(author)})
    return response

@app.route('/name/')
def index() -> str:
    name = request.args.get('name', default = '%', type=str)
    response=jsonify({'inventory': titlesearch(name)})
    return response 

@app.route('/ISBN/')
def isbn() -> str:
    isbn = request.args.get('isbn', default = '%', type=str)
    response=jsonify({'inventory': isbnsearch(isbn)})
    return response

@app.route('/info/')
def info() -> str:
    isbn = request.args.get('isbn', default = '%', type=str)
    response = jsonify(get_book_info(isbn))
    return response

@app.route('/available/')
def available() -> str:
    isbn = request.args.get('isbn', default = '%', type=str)
    response = jsonify(get_available_books(isbn))
    return response

@app.route('/rented/')
def rented() -> str:
    user = request.args.get('user', default = '%', type=str)
    response = jsonify(get_rented_books(user))
    return response

@app.route('/get_authors/')
def get_authors() -> str:
    response = jsonify(get_all_authors())
    return response

@app.route('/get_publishers/')
def get_publishers() -> str:
    response = jsonify(get_all_publ())
    return response

@app.route('/states/')
def get_states() -> str:
    response = jsonify(get_all_states())
    return response

@app.route('/countries/')
def get_countries() -> str:
    response = jsonify(get_all_countries())
    return response
    
@app.route('/rent_book/')
def rent():
    inventory_id = request.args.get('id', type=str)
    user = request.args.get('user', type=str)
    response = jsonify(rent_book(inventory_id, user))
    return response

@app.route('/return/')
def return_b():
    inventory_id = request.args.get('id', type=str)
    response = jsonify(return_book(inventory_id))
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')
