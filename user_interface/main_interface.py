from tkinter import *
import create_model.predict_input as predict

root = Tk()

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)

input_data = []

OPTIONS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]


def select(selection):
    input_data.append(OPTIONS.index(selection))


variable = StringVar(root)
variable.set('-')

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
choose1 = OptionMenu(root, variable, *OPTIONS, command=select)
choose1.grid(row=4, column=1, pady=10, padx=10, sticky=W)


def click():
    input_data.append(entry1.get())
    input_data.append(entry2.get())
    input_data.append(entry3.get())
    input_data.append(entry4.get())
    prediction = predict.predict_output(input_data)
    new_root = Tk()
    title = Label(new_root, text='Most likely you company will').grid(row=0, pady=10, padx=5)
    if prediction == 0:
        string = 'be closed'
    elif prediction == 1:
        string = 'be operating or will be sold'
    else:
        string = 'go to IPO'
    label8 = Label(new_root, text=string).grid(row=1, pady=10, padx=5)
    new_root.mainloop()


button1 = Button(root, text='Process data', command=click).grid(sticky=SE, row=5, column=1, pady=10, padx=10)

root.mainloop()
