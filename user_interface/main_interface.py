from tkinter import *

root = Tk()

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)


def callback(selection):
    print(selection)


OPTIONS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]

variable = StringVar(root)
variable.set(OPTIONS[0])

label1 = Label(root, text='Input expected size of funding (in millions): ').grid(row=0, pady=10, padx=5)
label2 = Label(root, text='Input expected size of funding rounds: ').grid(row=1, pady=10, padx=5)
label3 = Label(root, text='Input date of foundation of the company (in format Y-m-d): ').grid(row=2, pady=10, padx=5)
label4 = Label(root, text='Input the category of the company: ').grid(row=3, pady=10, padx=5)
label5 = Label(root, text='Choose your region: ').grid(row=4, pady=10, padx=5)

entry1 = Entry(root, width=50)
entry1.grid(row=0, column=1, pady=10, padx=10, sticky=E)
entry2 = Entry(root, width=50)
entry2.grid(row=1, column=1, pady=10, padx=10, sticky=E)
entry3 = Entry(root, width=50)
entry3.grid(row=2, column=1, pady=10, padx=10, sticky=E)
entry4 = Entry(root, width=50)
entry4.grid(row=3, column=1, pady=10, padx=10, sticky=E)
entry5 = OptionMenu(root, variable, *OPTIONS, command=callback)
entry5.grid(row=4, column=1, pady=10, padx=10, sticky=W)


def click():
    print(entry1.get())
    print(entry2.get())
    print(entry3.get())
    print(entry4.get())


button1 = Button(root, text='Process data', command=click).grid(sticky=SE, row=5, column=1, pady=10, padx=10)

root.mainloop()
