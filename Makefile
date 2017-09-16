.PHONY: 

all: clean static

static: .PHONY clean
	python main.py
	open ./imgs

clean:
	-rm imgs/*.png
	-rm *.gif

gif: .PHONY clean backup
	python main.py -gif
	open ./

backup:
	-gcp --backup=t anim.gif ~/Desktop/
