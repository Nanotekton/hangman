# Closed-loop optimization of general conditions for heteroaryl Suzuki coupling 
## Nicholas H. Angello, Vandana Rathore, Wiktor Beker, Agnieszka Wołos, Edward R. Jira, Rafał Roszak, Tony C. Wu, Alán Aspuru-Guzik, Charles M. Schroeder, Bartosz A. Grzybowski, Martin D. Burke

``do_step.py`` - script that a) downloads the speadsheet from shared Google drive, b) trains the model and makes prediction for the next step and c) uploads next reactions to the shared Google drive

The script requires a dictionary placed in JSON file cache.json, which have to provide a field ``url`` pointing to the spreadsheet.

The `update_gdirve` option requires a properly configured `rclone`.
