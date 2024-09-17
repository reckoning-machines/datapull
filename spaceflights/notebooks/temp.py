# array(['AXP US', 'BAC US', 'BK US', 'C US', 'GS US', 'JPM US', 'MS US',
#       'PNC US', 'STT US', 'SYF US', 'USB US', 'WFC US'], dtype=object)
# from zipfile import ZipFile

# original_zip = ZipFile ('betas_table.csv.zip', 'r')
# new_zip = ZipFile ('betas_table.zip', 'w')
# for item in original_zip.infolist():
#    buffer = original_zip.read(item.filename)
#    if not str(item.filename).startswith('__MACOSX/'):
#      new_zip.writestr(item, buffer)

# new_zip.close()
# original_zip.close()
