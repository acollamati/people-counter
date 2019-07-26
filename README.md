# People Counter based on Face Recognition

It records when a person's face is recognised:

* If no similar faces have been encoded don't record. Wait for at least two similar faces to be encoded. This means people in records are actually them, and not duplicates
* Once a face is recorded, its encoding is saved and re-used in the future
* A set of "famous people" images can be used to detect them (put the images in a directory `./images` with name=name of the person and pictures inside it.