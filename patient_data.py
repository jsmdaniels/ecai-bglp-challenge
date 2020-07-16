# -*- coding: utf-8 -*-

"""
Created on Mon Jan 20 2020

@author: Sadegh Mirshekarian

@edited: John Daniels

Processing pipeline for OhioT1DM dataset
"""

import sys
import xml.etree.cElementTree as XMLParser
import pandas as pd
import scipy.io as sio
import numpy as np
import os
import math
import csv
import time
import datetime as dt
import matplotlib.pyplot as plt

#In[2]:
cats = ['timeOfDay_real', 'bgl', 'bolus', 'meal', 'finger_stick', 'basal', 'temp_basal', 'exercise', 'work', 'stressors', 'hypo_event', 'sleep','illness',
        'basis_heart_rate', 'basis_gsr', 'basis_skin_temperature', 'basis_air_temperature', 'basis_steps', 'basis_sleep', 'basis_steps', 'acceleration']

# cats = ['bgl', 'finger_stick', 'meal', 'bolus']

def load_from_xml(fileNamesAndIDs, res=5, includeCats=cats, dtFormat='%d-%m-%Y %H:%M:%S', verbose=True, train = 0):
  """ Loads patient data from an XML file containing all the bgl and even information.
  The first category in the file must be BGL values (anything before bgl will be skipped)
  because other catogires might depend on them. The other categories can be in any order.
  NOTE: hypo action should occur after meals in the input file, also temp basal after basal
  NOTE: when 'meal' is included in the categories, hypo_actions are considered meals
  TAKES:
      fileNamesAndIDs: a list of pairs of (patiend ID, filename) to process
      res: resolution in minutes of the output sequences. Default is 5 minutes for
          blood glucose measurements.
      includeCats: include these categories. It must be a list of categories.
          Note that 'bgl' is always automatically added.
      dtFormat: expected date time format in the XML file.
  RETURNS:
      data: a dictionary of {pID:X} where X is an N x D list, N: number of bgl data and
      timeStamps: the time stamps corresponding to BGL data
      isRealBGL: a dict of binary lists indicating whether each sample has a real BGL or not (an interpolated one)
  """
  data = {}
  data_label = {}
  timeStamps = {}
  isRealBGL = {}

  def _valueAtTimeInCat(tm, category, threshold):
    """
    Helper function that searches in a 'category' (which is just a dictionary of
    datetime:value pairs) and checks if 'tm' is in the keys within a 'threshold'.
    """
    delta1 = dt.timedelta(minutes=1)
    for r in range(threshold+1):
        if (tm + r*delta1) in category:
            return category[tm + r*delta1]
        elif (tm - r*delta1) in category:
            return category[tm - r*delta1]
    return 0

  if not isinstance(includeCats, list):
      raise Exception('The category list to include must be a LIST of category labels.')


  for pID, fname in fileNamesAndIDs:
      # print("\n> Parsing file {} for patient '{}'...".format(fname, pID))
      xmlRoot = XMLParser.parse(fname).getroot()
      xmlIter = xmlRoot.iter()

      item = next(xmlIter)
      while item.tag != 'glucose_level':
          item = next(xmlIter)

      # print(" > Parsing category 'bgl'...")
      columns         = []
      categories      = [] # list of encountered categories
      bgl             = []
      isRealBGL[pID]  = []
      categories.append(bgl)
      columns.append('G')
      timeStamps[pID] = []
      prevBGLarray    = []

      item = next(xmlIter)
      prevLevel = 0
      prevDT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)

      while item.tag == 'event':
          curDT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
          curLvl = float(item.attrib['value'])
          diff = (curDT - prevDT).total_seconds() / 60

          if diff > res:
              if verbose:
                  print(" Warning: bgl data discontinuity at {}".format(prevDT))
              s = int(diff // res - (diff % res == 0))
              m = (curLvl - prevLevel) / diff

              for i in range(s):
                  prevLevel += m * res

                  bgl.append(prevLevel)
                  isRealBGL[pID].append(False)
                  timeStamps[pID].append(prevDT + dt.timedelta(minutes=res*(i+1)))

          bgl.append(curLvl)
          # if(len())
          prevBGLarray = bgl
          isRealBGL[pID].append(True)
          timeStamps[pID].append(curDT)

          prevDT = curDT
          prevLevel = curLvl
          item = next(xmlIter)

      if 'timeOfDay_real' in includeCats:
          #print(" > Time of day is being included as total seconds since midnight...")
          timeOfDay_real = {ts:(ts-ts.replace(minute=0, hour=0)).total_seconds() for ts in timeStamps[pID]}
          categories.append(timeOfDay_real)
          columns.append('ToD')

      if 'timeOfDay_ticks' in includeCats:
          #print(" > Time of day is being included as one feature with ticks on the hour...")
          timeOfDay_ticks = {}
          prevHour = -1
          for ts in timeStamps[pID]:
              thisHour = ts.hour
              timeOfDay_ticks[ts] = (thisHour != prevHour)*1
              prevHour = ts.hour

          categories.append(timeOfDay_ticks)

      while item.tag:
          try:
              if item.tag=='finger_stick' and 'finger_stick' in includeCats:
                  #print(" > Parsing category 'finger_stick'...")
                  fstick = {}

                  categories.append(fstick)
                  columns.append('CAL')
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      fstick[DT] = float(item.attrib['value'])
                      item = next(xmlIter)

              elif item.tag=='basal' and 'basal' in includeCats:
                  #print(" > Parsing category 'basal'...")
                  basal = {}
                  # columns.append('BAS')
                  categories.append(basal)
                  columns.append('BAS')
                  beginDT = None
                  item = next(xmlIter)
                  _value = None
                  while item.tag=='event':
                      endDT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      if beginDT:
                          while beginDT <= endDT:
                              basal[beginDT] = _value
                              beginDT = beginDT + dt.timedelta(minutes=1)
                      beginDT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      _value = item.attrib['value']
                      item = next(xmlIter)

              elif item.tag=='temp_basal' and 'basal' in includeCats:
                  #print(" > Parsing category 'temp_basal' and correcting 'basal'...")
                  item = next(xmlIter)
                  while item.tag=='event':
                      beginDT = dt.datetime.strptime(item.attrib['ts_begin'], dtFormat).replace(second=0)
                      endDT = dt.datetime.strptime(item.attrib['ts_end'], dtFormat).replace(second=0)
                      while beginDT <= endDT:
                          basal[beginDT] = item.attrib['value']
                          beginDT = beginDT + dt.timedelta(minutes=1)
                      item = next(xmlIter)

              elif item.tag=='bolus' and 'bolus' in includeCats:
                  #print(" > Parsing category 'bolus'...")
                  # columns.append('I')
                  bolus = {}
                  categories.append(bolus)
                  columns.append('I')
                  item = next(xmlIter)
                  while item.tag=='event':
                      beginDT = dt.datetime.strptime(item.attrib['ts_begin'], dtFormat).replace(second=0)
                      endDT = dt.datetime.strptime(item.attrib['ts_end'], dtFormat).replace(second=0)
                      num_minutes = max(1, (endDT-beginDT).total_seconds() / 60)
                      bolus_per_minute = float(item.attrib['dose']) / num_minutes
                      while beginDT <= endDT:
                          bolus[beginDT] = min(res, num_minutes) * bolus_per_minute
                          beginDT = beginDT + dt.timedelta(minutes=res)
                          num_minutes = (endDT-beginDT).total_seconds() / 60
                      item = next(xmlIter)

              elif item.tag=='meal' and 'meal' in includeCats:
                  #print(" > Parsing category 'meal'...")
                  meal = {}
                  # columns.append('M')
                  categories.append(meal)
                  columns.append('M')
                  item = next(xmlIter)
                  while item.tag=='event' or item.tag=='food':
                      if item.tag=='event':
                          DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                          carb = float(item.attrib['carbs'])
                          meal[DT] = carb
                      item = next(xmlIter)

              elif item.tag=='sleep' and 'sleep' in includeCats:
                  #print(" > Parsing category 'sleep'...")
                  sleep = {}
                  # columns.append('sleep')
                  categories.append(sleep)
                  columns.append('sleep')
                  item = next(xmlIter)
                  while item.tag=='event':
                      beginDT = dt.datetime.strptime(item.attrib['ts_end'], dtFormat).replace(second=0)
                      endDT = dt.datetime.strptime(item.attrib['ts_begin'], dtFormat).replace(second=0)
                      if beginDT > endDT:
                          beginDT, endDT = endDT, beginDT
                      if endDT-beginDT > dt.timedelta(hours=12) and verbose:
                          print(" Warning: long sleep detected from {} to {}, duration={} hr"
                                  .format(beginDT, endDT, (endDT-beginDT).total_seconds()/3600))
                      while beginDT <= endDT:
                          sleep[beginDT] = 1
                          beginDT = beginDT + dt.timedelta(minutes=1)
                      item = next(xmlIter)

              elif item.tag=='work' and 'work' in includeCats:
                  #print(" > Parsing category 'work'...")
                  work = {}
                  # columns.append('work')
                  categories.append(work)
                  columns.append('work')
                  item = next(xmlIter)
                  while item.tag=='event':
                      beginDT = dt.datetime.strptime(item.attrib['ts_begin'], dtFormat).replace(second=0)
                      endDT = dt.datetime.strptime(item.attrib['ts_end'], dtFormat).replace(second=0)
                      if endDT-beginDT > dt.timedelta(hours=12) and verbose:
                          print(" Warning: long work detected from {} to {}, duration={} hr"
                                  .format(beginDT, endDT, (endDT-beginDT).total_seconds()/3600))
                      while beginDT <= endDT:
                          work[beginDT] = float(item.attrib['intensity'])
                          beginDT = beginDT + dt.timedelta(minutes=1)
                      item = next(xmlIter)

              elif item.tag=='infusion_set' and 'infusion_set' in includeCats:
                  #print(" > Parsing category 'infusion_set'...")
                  infusion_set = {}
                  # columns.append('infusion_set')
                  categories.append(infusion_set)
                  columns.append('infusion_set')
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      infusion_set[DT] = 1
                      item = next(xmlIter)

              elif item.tag=='hypo_event' and 'hypo_event' in includeCats:
                  #print(" > Parsing category 'hypo_event'...")
                  hypo_event = {}
                  # columns.append('hypo_event')
                  categories.append(hypo_event)
                  columns.append('hypo_event')
                  item = next(xmlIter)
                  while item.tag=='event' or item.tag=='symptom':
                      if item.tag=='event':
                          DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                          hypo_event[DT] = 1
                      item = next(xmlIter)

              elif item.tag=='hypo_action' and 'meal' in includeCats:
                  #print(" > Parsing category hypo_action and adding to 'meal...")
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      if DT in meal and verbose:
                          print(" Warning: hypo action already in 'meals' at {} with carbs={}, replacing with {}..."
                                  .format(DT, meal[DT], item.attrib['carbs']))
                      meal[DT] = item.attrib['carbs']
                      item = next(xmlIter)

              elif item.tag=='exercise' and 'exercise' in includeCats:
                  #print(" > Parsing category 'exercise'...")
                  exercise = {}
                  # columns.append('exercise')
                  categories.append(exercise)
                  columns.append('exercise')
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      duration = float(item.attrib['duration'])
                      intensity = float(item.attrib['intensity'])
                      for i in range(int(duration)+1):
                          exercise[DT+dt.timedelta(minutes=i)] = intensity
                      item = next(xmlIter)

              elif item.tag=='basis_heart_rate' and 'basis_heart_rate' in includeCats:
                  #print(" > Parsing category 'basis_heart_rate'...")
                  hrate = {}
                  # columns.append('HR')
                  categories.append(hrate)
                  columns.append('HR')
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      hrate[DT] = float(item.attrib['value'])
                      item = next(xmlIter)

              elif item.tag=='basis_gsr' and 'basis_gsr' in includeCats:
                  #print(" > Parsing category 'basis_gsr'...")
                  gsr = {}
                  # columns.append('GSR')
                  categories.append(gsr)
                  columns.append('GSR')
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      gsr[DT] = float(item.attrib['value'])
                      item = next(xmlIter)

              elif item.tag=='basis_skin_temperature' and 'basis_skin_temperature' in includeCats:
                  #print(" > Parsing category 'basis_skin_temperature'...")
                  skin_temp = {}
                  # columns.append('ST')
                  categories.append(skin_temp)
                  columns.append('ST')
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      skin_temp[DT] = float(item.attrib['value'])
                      item = next(xmlIter)

              elif item.tag=='basis_air_temperature' and 'basis_air_temperature' in includeCats:
                  #print(" > Parsing category 'basis_air_temperature'...")
                  air_temp = {}
                  # columns.append('AT')
                  categories.append(air_temp)
                  columns.append('AT')
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      air_temp[DT] = float(item.attrib['value'])
                      item = next(xmlIter)

              elif item.tag=='basis_steps' and 'basis_steps' in includeCats:
                  #print(" > Parsing category 'basis_steps'...")
                  steps = {}
                  # columns.append('STP')
                  categories.append(steps)
                  columns.append('STP')
                  item = next(xmlIter)
                  while item.tag=='event':
                      DT = dt.datetime.strptime(item.attrib['ts'], dtFormat).replace(second=0)
                      steps[DT] = float(item.attrib['value'])
                      item = next(xmlIter)


              else:
                  item = next(xmlIter)

          except StopIteration:
              break

      #print(" > Parsing done, now merging the data into a single matrix...")

      N = len(bgl)
      D = len(categories)
      data[pID] = np.zeros((N, D))
      data_label[pID] = np.array(columns)

      for t, tm in enumerate(timeStamps[pID]):
          data[pID][t, 0] = bgl[t]
          for c, cat in enumerate(categories[1:]):
              data[pID][t, c+1] = _valueAtTimeInCat(tm, cat, threshold=2)

  return data, data_label, timeStamps, isRealBGL
