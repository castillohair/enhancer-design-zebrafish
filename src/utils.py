
def sanitize_cell_state(s):
    if s=='olfactory/adenohypophyseal placode(6somite)':
        s = 'olfactory or adenohypophyseal placode(6somite)'
    return s

def unsanitize_cell_state(s):
    if s=='olfactory or adenohypophyseal placode(6somite)':
        s = 'olfactory/adenohypophyseal placode(6somite)'
    return s
