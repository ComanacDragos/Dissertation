import tkinter as tk
from tkinter import (Tk)

from PIL import ImageTk, Image

from backend.enums import Stage
from backend.visualizer.object_detection_service import ObjectDetectionService


class Visualizer:
    def __init__(self, service):
        self.service = service
        self.window = Tk()
        self.window.title("Image visualizer")

        self.listbox = tk.Listbox(self.window, width=20, height=10)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar = tk.Scrollbar(self.window, orient=tk.VERTICAL, command=self.listbox.yview)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.config(yscrollcommand=self.scrollbar.set)

        for name in self.service.names:
            self.listbox.insert(tk.END, name)

        self.listbox.bind("<<ListboxSelect>>", self.listbox_trigger)

        self.window.geometry('1280x800')
        self.window.update()
        self.show_image(0)
        self.listbox.focus()

        self.window.mainloop()

    def listbox_trigger(self, event):
        selection = self.listbox.curselection()
        if selection:
            index = int(selection[0])
            self.show_image(index)

    def show_image(self, index):
        img = Image.fromarray(self.service[index])

        img_aspect_ratio = img.height / img.width

        new_width = int(self.window.winfo_width() * 0.75)
        new_height = int(img_aspect_ratio * new_width)

        img = img.resize((new_width, new_height))
        img = ImageTk.PhotoImage(img)
        if hasattr(self, "image_label"):
            self.image_label.configure(image=img)
            self.image_label.image = img
        else:
            self.image_label = tk.Label(self.window, image=img)
            self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.image_label.image = img


if __name__ == '__main__':
    from config.kitti_object_detection.data_generator_config import KittiDataGeneratorConfig

    KittiDataGeneratorConfig.BATCH_SIZE = 1
    KittiDataGeneratorConfig.AUGMENTATIONS = None
    KittiDataGeneratorConfig.SHUFFLE = False
    Visualizer(
        service=ObjectDetectionService(KittiDataGeneratorConfig.build(Stage.VAL))
    )
