"""This thing finds/removes duplicates from a folder. Can search across multiple
folders, which is nice. """

import os
import sys
import hashlib

# Function to calculate the MD5 has of a given file. Takes path, returns HEX.

def hashfile(path, blocksize= 65536):
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()

#Function to scan directory for duplicated files:

def findDup(parentFolder):
    #Dups in format {hash:[names]}
    dups = {}
    for dirName, subdirs, fileList in os.walk(parentFolder):
        print('Scanning %s...' % dirName)
        for filename in fileList:
            #Get the path to the file
            path = os.path.join(dirName, filename)
            #Calculate hash
            file_hash = hashfile(path)
            #Add or append the file path
            if file_hash in dups:
                dups[file_hash].append(path)
            else:
                dups[file_hash] = [path]
    return dups

# Function to join two dictionaries
"""Takes two dictionaries, iterates over the second one and checks if the
key exists in the first dict. if it does, appends the values in the second dict to the ones in the 
first dict. if the key does not exist, it stores it in the first dict. At the end, the first dict has all info. """
def joinDicts(dict1, dict2):
    for key in dict2.keys():
        if key in dict1:
            dict1[key] = dict1[key] + dict2[key]
        else:
            dict1[key] = dict2[key]



#Printing results

def printResults(dict1):
    results = list(filter(lambda x: len(x) >1, dict1.values()))
    if len(results) > 0:
        print('Duplicates Found: ')
        print('The following files are identical, The name could differ but the content is identical')
        print('__________________________')
        for result in results:
            print('this is result: %s '%result)
            print('thi is len result: %s'%len(result))
            # result_to_delete = results[0]
            # print('result to delete: %s' %result_to_delete)
            for subresult in result[:-1]:
                print('\t\t%s , (subresult)' %subresult)
                # os.remove(subresult)    #Uncomment to remove duplicates, comment again to see if all have been removed
            print('_________________________')
    else:
        print('No duplicate files found.')

#
# if __name__ == '__main__':
#     if len(sys.argv) >1:
#         dups = {}
#         folders = sys.argv[1:]
#         for i in folders:
#             #Iterate the folders given
#             if os.path.exists(i):
#                 #Find the duplicated files and append them to the dups
#                 joinDicts(dups, findDup(i))
#             else:
#                 print('%s is not a valid path, please verify' % i)
#                 sys.exit()
#         printResults(dups)
#     else:
#         print('Usage: python dupFinder.py folder or python dupFinder.py folder 1 folder 2 folder 3 ')


a = findDup('/Users/mk/PycharmProjects/AI/TFPOETS/Test/Grey')
printResults(a)


