###############################################################################
# a gui to show a specify group information
###############################################################################

import argparse
import os
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from focus_measure_types import FOCUS_MEASURE_TYPES, FOCUS_MEASURE_NAMES
from group_utils import *


class Application(tk.Frame):
    def __init__(self, master=None, load_path=None):
        super().__init__(master)
        self.master = master
        # self.master.minize((800,600))
        self.pack(expand=True, fill=tk.BOTH)
        self.group = None
        self.current_pos = None
        self.fms = []
        self.load_path = load_path
        self.init_view()
        self.init_data()
        self.update_info()

    def init_view(self):
        self.scrollbar_positions = tk.Scrollbar(self)
        self.listbox_positions = tk.Listbox(self, selectbackground="#b0e0e6", selectmode=tk.SINGLE, yscrollcommand=self.scrollbar_positions.set, width=10)
        self.listbox_positions.bind("<Double-Button-1>", self.select_position)
        self.listbox_positions.pack(side=tk.LEFT, fill=tk.Y)
        self.scrollbar_positions.pack(side=tk.LEFT, fill=tk.Y)
        self.scrollbar_positions.config(command=self.listbox_positions.yview)

        info_frame = tk.Frame(self)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.label_group_name = tk.Label(info_frame, text="")
        self.label_group_name.pack()

        frame1 = tk.Frame(info_frame)
        frame1.pack()
        label1 = tk.Label(frame1, text="peak position index")
        label1.pack(side=tk.LEFT)
        self.peak_pos_idx = tk.StringVar()
        self.entry_peak_pos_idx = tk.Entry(frame1, bd=5, textvariable=self.peak_pos_idx)
        self.entry_peak_pos_idx.pack(side=tk.LEFT)
        self.botton_peak_pos_idx = tk.Button(frame1, text="change", command=self.edit_peak_pos_idx)
        self.botton_peak_pos_idx.pack(side=tk.LEFT)

        self.listbox_focus_measures = tk.Listbox(info_frame, selectbackground="#b0e0e6", selectmode=tk.SINGLE, width=50)
        self.listbox_focus_measures.pack(expand=True, fill=tk.Y)
        self.listbox_focus_measures.bind("<Double-Button-1>", self.select_focus_measure)

        center_frame = tk.Frame(self)
        center_frame.pack(fill=tk.BOTH)
        self.label_image = tk.Label(center_frame)
        self.label_image.pack()

    def init_data(self):
        self.group = load_group_json(self.load_path)
        self.label_group_name["text"] = "%s\n%d position(s)" % (self.group.name, self.group.pos_number)
        self.fms = []
        for i, p in enumerate(self.group.positions):
            self.listbox_positions.insert(tk.END, "%3d-%5d" % (p.pos_idx, p.z_coordinate))
            pos_fms = []
            for i in range(1, len(FOCUS_MEASURE_TYPES)):
                pos_fms.append(p.focus_measures.get(FOCUS_MEASURE_TYPES[i], -1))
            self.fms.append(pos_fms)
        self.fms = np.array(self.fms)
        argmaxs = np.argmax(self.fms, axis=0)
        maxs = np.max(self.fms, axis=0)
        for i in range(len(FOCUS_MEASURE_TYPES) - 1):
            type = FOCUS_MEASURE_TYPES[i+1]
            if maxs[i] == -1:
                self.listbox_focus_measures.insert(tk.END, "%26s z: %5s index: %3s" % (type, "", ""))
            else:
                self.listbox_focus_measures.insert(tk.END, "%26s z: %6d index: %3d" % (type, self.group.positions[argmaxs[i]].z_coordinate, argmaxs[i]))
        self.current_pos = self.group.positions[0]

    def show_image(self, position):
        image_path = os.path.join(position.dirname, position.filename)
        image = Image.open(image_path)
        # image.thumbnail((1200, 1200))
        image = ImageTk.PhotoImage(image)
        self.label_image["image"] = image
        self.image = image

    def update_info(self):
        self.show_image(self.current_pos)

    def select_position(self, event):
        idxs = self.listbox_positions.curselection()
        if len(idxs) == 0:
            return
        idx = idxs[0]
        self.current_pos = self.group.positions[idx]
        self.update_info()

    def select_focus_measure(self, event):
        idxs = self.listbox_focus_measures.curselection()
        if len(idxs) == 0:
            return
        idx = idxs[0]
        type = FOCUS_MEASURE_NAMES[idx + 1]
        fms = self.fms[:, idx]
        zs = []
        for p in self.group.positions:
            zs.append(p.z_coordinate)
        plt.plot(zs, fms, label=type)
        plt.legend(loc="best")
        plt.show()

    def edit_peak_pos_idx(self):
        idx = self.peak_pos_idx.get()
        try:
            idx = int(idx)
            if idx == -1:
                self.group.pos_peak_idx = -1
                self.peak_pos_idx.set("")
            elif 0 <= idx < len(self.group.positions):
                self.group.pos_peak_idx = idx
            else:
                self.peak_pos_idx.set("")
        except Exception:
            self.peak_pos_idx.set("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("load_path", help="the path to load dataset object")
    args = parser.parse_args()

    root = tk.Tk()
    app = Application(master=root, load_path=args.load_path)
    # root.protocol("WM_DELETE_WINDOW", app.close)
    app.mainloop()
