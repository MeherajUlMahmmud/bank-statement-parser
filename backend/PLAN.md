my plan with this project:
 - this project is about converting PDF bank statements into csv files
 - users will upload PDF bank statements and from the pdf file images will be extracted
 - those images will be OCRed by OlamOCR
 - then an agent will cleanup the raw ocr response 
 - then the cleanedup ocr response will be given to another agent to convert the json into structured JSON data.
 - the JSOn data then will be converted into formatted CSv data
 - in this whole pipeline, everything will be saved to database.