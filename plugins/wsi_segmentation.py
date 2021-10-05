import SlideRunner.general.SlideRunnerPlugin as SlideRunnerPlugin
from threading import Thread
from queue import Queue
import cv2
import os
import numpy as np
import h5py
import openslide


class Plugin(SlideRunnerPlugin.SlideRunnerPlugin):
    version = 0.1
    shortName = 'WSI Segmentation Overlay'
    inQueue = Queue()
    outQueue = Queue()
    initialOpacity = 0.6
    updateTimer = 0.1
    outputType = SlideRunnerPlugin.PluginOutputType.RGB_OVERLAY
    description = 'Visualize segmentation results'
    pluginType = SlideRunnerPlugin.PluginTypes.WHOLESLIDE_PLUGIN
    configurationList = list((
        SlideRunnerPlugin.FilePickerConfigurationEntry(uid='file', name='Result file', mask='*.hdf5'),
    ))


    COLORS = [[255, 255, 255, 255], # BG
              [0, 0, 255, 255], # Dermis
              [0, 255, 255, 255], # Epidermis
              [255, 0, 0, 255], # Subcutis
              [255, 20, 147, 255], # Inflamm/Necrosis
              [255, 128, 0, 255]] # Tumor



    def __init__(self, statusQueue: Queue):
        self.statusQueue = statusQueue
        self.p = Thread(target=self.queueWorker, daemon=True)
        self.p.start()
        pass

    def queueWorker(self):
        quitSignal = False
        oldArchive = ''
        oldSlide = ''
        oldCoordinates = [-1, -1, -1, -1]
        while not quitSignal:
            job = SlideRunnerPlugin.pluginJob(self.inQueue.get())
            print(job)
            print(job.configuration)

            if (job.jobDescription == SlideRunnerPlugin.JobDescription.QUIT_PLUGIN_THREAD):
                # signal to exit this thread
                quitSignal = True
                continue

            if (job.configuration['file'] == oldArchive) and (job.slideFilename == oldSlide) and np.all(
                    job.coordinates == oldCoordinates):
                continue

            if not (os.path.exists(job.configuration['file'])):
                continue

            fileChanged = job.configuration['file'] != oldArchive
            slideChanged = job.slideFilename != oldSlide

            oldArchive = job.configuration['file']
            oldSlide = job.slideFilename
            oldCoordinates = job.coordinates

            self.slideObj = openslide.open_slide(job.slideFilename)

            if (fileChanged):
                self.resultsArchive = h5py.File(oldArchive, "r")
                self.factor = 64
                self.ds = int(np.round(self.factor / (self.slideObj.dimensions[0] / self.resultsArchive["segmentation"].shape[1])))
                self.downsampledMap = self.resultsArchive["segmentation"][::self.ds,::self.ds]

                self.scaleX = ((job.coordinates[2]) / job.currentImage.shape[1])
                self.scaleY = ((job.coordinates[3]) / job.currentImage.shape[0])
                print('Opened results container.')
                print('Downsampled image: ', self.downsampledMap.shape)


            if slideChanged:
                self.slideObj = openslide.open_slide(job.slideFilename)


            print('returning overlay...')
            current_shape = np.array(job.currentImage).shape
            coords_overlay = np.int16(np.array(job.coordinates)/self.factor)
            image = self.downsampledMap[
                    max(coords_overlay[1], 0):min(coords_overlay[1] + coords_overlay[3], self.downsampledMap.shape[0]),
                    max(coords_overlay[0], 0):min(coords_overlay[0] + coords_overlay[2],self.downsampledMap.shape[1])]
            image = np.asarray([self.COLORS[i] for i in np.int16(image.flatten())],dtype=np.uint8).reshape((image.shape[0], image.shape[1], -1))
            image = cv2.copyMakeBorder(image,np.abs(min(0, coords_overlay[1])),0,np.abs(min(0, coords_overlay[0])),0,cv2.BORDER_CONSTANT,value=(0,0,0,255))
            image = cv2.copyMakeBorder(image,0,max(0,coords_overlay[3]-image.shape[0]),0,max(0,coords_overlay[2]-image.shape[1]),cv2.BORDER_CONSTANT,value=(0,0,0,255))
            image = cv2.resize(image, dsize=(current_shape[1], current_shape[0]))

            self.returnImage(np.float32(image[:,:,0:3]))
