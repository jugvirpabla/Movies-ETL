# Movies-ETL

## Project Overview
To create an automated pipeline that takes in new data, performs the appropriate transformations, and loads the data into existing tables.

## Resources
 - Data Sources: wikipedia.movies.json, movies_metadata.csv, ratings.csv
 - Software: Jupyter Notebooks, Python, Pandas, Numpy, Sqlalchemy, PostgreSQL, Visual Studio Code

## Limitations
 - Dividing the large file to upload in chucks could be an issue as the file grows and grows in size.
 - There has to be someone checking the data regularly, other wise there could be errors that go unnoticed, i.e. budget numbers.
 - Wikipedia may not have accurate information.
 - Is the JSON file being updated as new infromation comes avalible for older movies?
 - There could be corrupted data, as we saw earlier, two movies being mixed together.
