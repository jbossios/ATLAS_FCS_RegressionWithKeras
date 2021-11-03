#########################################################################################################
# Name:    MakeHTMLpage.py                                                                              #
# Purpose: Make HTML page with all plots comparing true vs predicted extrapolation weight distributions #
# Author:  Jona Bossio (jbossios@cern.ch)                                                               #
# Usage:   python MakeHTMLpage.py                                                                       #
#########################################################################################################

Pages = {
#  'v17_photons'   : {'Particle':'photons',   'Version':'v17'},
#  'v17_electrons' : {'Particle':'electrons', 'Version':'v17'},
#  'v18_pions'     : {'Particle':'pions',     'Version':'v18'},
#  'v18_electrons' : {'Particle':'electrons', 'Version':'v18'},
#  'v09_electrons' : {'Particle':'electrons', 'Version':'v09'},
#  'v10_pions'     : {'Particle':'pions',     'Version':'v10'},
}

Msg = { # particles used in training
  'v09' : 'electrons',
  'v10' : 'pions',
  'v17' : 'photons and electrons',
  'v18' : 'pions and electrons',
}

# PATH were pages will be created
outPATH = '/eos/user/j/jbossios/www/FCS/RegressionResultsPages/'

############################################################################################################
# DO NOT MODIFY (below this line)
############################################################################################################

import os

# Loop over pages to create
for PageName,Dict in Pages.items():
  Particle = Dict['Particle']
  # Path to input files
  inPATH = 'Plots/{}/{}/PNG/'.format(Particle,Dict['Version'])
  # Copy PNGs to www/
  print('INFO: Copying files to www dir...')
  os.system('mkdir {}/RegressionResults/{}/'.format(outPATH,PageName))
  os.system('cp {}*.png {}/RegressionResults/{}/'.format(inPATH,outPATH,PageName))
  print('INFO: Done copying')
  # List of layers
  Layers = [0,1,2,3,12]
  if Particle == 'pions': Layers += [13,14]
  # Define eta bins
  if Particle == 'photons' or Particle == 'electrons':
    EtaBins = ['{}_{}'.format(x*5,x*5+5) for x in range(26)]
  elif Particle == 'pions':
    EtaBins = ['{}_{}'.format(x*5,x*5+5) for x in range(16)]
  # Define energy bins
  EBins = [1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304]
  # Create page
  print('INFO: Creating {} page'.format(outPATH+PageName+'.html'))
  ofile       = open(outPATH+PageName+'.html',"w")
  ofile.write("<!DOCTYPE html>\n")
  ofile.write("<html>\n")
  ofile.write("<head>\n")
  ofile.write("<title>Results for "+PageName+"</title>\n")
  ofile.write("</head>\n")
  ofile.write("<body>\n")
  # Print message
  ofile.write('<center><p>Results for {} using NN trained with {}</p></center>'.format(Particle,Msg[Dict['Version']]))
  # Create table with headers
  ofile.write("<table style:'width:100%'>")
  ofile.write(" <tr>\n")
  for layer in Layers:
    ofile.write("   <th>Layer "+str(layer)+"</th>\n")
  ofile.write(" </tr>")
  # Loop over eta bins
  for etaBin in EtaBins:
    print('INFO: Doing EtaBin = {}'.format(etaBin))
    # Loop over energy bins
    for eBin in EBins:
      # Show eta and energy bin
      ofile.write(" <tr>")
      for ilayer in Layers:
        ofile.write("   <td><center><p>Eta={}, E={}</p></center>\n".format(etaBin,eBin))
      ofile.write(" </tr>\n")
      ofile.write(" <tr>")
      # Loop over layers
      for ilayer in Layers:
        FileName = 'eta{0}_E{1}_Real_relu_{2}_{0}_TruePredictionDiff_extrapWeight_{3}.png'.format(etaBin,eBin,Particle,ilayer)
        width    = 1800/len(Layers)
        height   = width - 50
        ofile.write("   <td><img src='RegressionResults/{0}/{1}' width='{2}' height='{3}'></td>\n".format(PageName,FileName,width,height))
      ofile.write(" </tr>\n")
  ofile.write("</table>")
  ofile.write("</body>\n")
  ofile.write("</html>\n")
  ofile.close()

print('>>> ALL DONE <<<')
