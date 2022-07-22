# Software Engineering Project
## Samuel Steiner, Brian Mckenna, Jean-Paul Michaud Sun Tae Park, Mia Salvati

This was an open-ended project done for CSC-445 Software Engineering class at Eastern Connecticut State University.

The goal of the project was to create a piece of software which resembled a Library Management System. This was accomplished using a microservice architecture using Vue.js as the frontend and Flask and MySQL for the backend. 

To see the project in action just clone the repository locally and run 
```
docker-compose build
docker-compose up
```

after letting the docker containers start up you can register for an account on the software to set the account to a admin login to the keycloak server with the username and password set in the compose file  the default is below  and the port is also below
```
User: admin 
Pass: admin

localhost:8090
``` 

You can then go to users and set the new user realm-role to librarian.

Currently the add_book function does not work, and form validation is not implemented in vue these features were unable to be added due to time. 

Some shortcuts were also made due to time to showcase some features but in a real production environment you would not allow these to make it to distribution. These include not having the backend also secured by keycloak, having no form validation when SQL queries are done to prevent SQL injection and other SQL security measures.  You would also not use the default H2 database for keycloak. 

Things which I would have liked to add include an event broker to handle cross microservice interactions. Adding more microservices such as one for DVDs or a transaction microservice.
