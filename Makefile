
all: backup clean static gif

static:
	python main.py
	open ./output

clean:
	-rm output/*.png
	-rm *.gif

gif:
	python make_gif.py
	open ./

sgif: clean static gif

backup:
	-gcp --backup=t anim.gif ~/Desktop/
