import commune

def delete_all_files(data = {'bro': 2200}):
    self = commune.Module()
    self.put_json('bro/fam.json', data=data)
    self.put_json('bro/dawg', data=data)
    self.rm_json('**')
    assert len(self.glob('**')) == 0, self.glob('**')
    
def delete_individual_files(data = {'bro': 2200}):
    self = commune.Module()
    self.put_json('bro/fam.json', data=data)
    self.put_json('bro/dawg', data=data)
    assert len(self.glob('**')) == 2, self.glob('**')
    self.rm_json('bro/fam')
    assert len(self.glob('**')) == 1, len(self.glob('**'))
    self.rm_json('bro/dawg.json')
    assert len(self.glob('**')) == 0
    
def delete_directory(data = {'bro': 2200}):
    self = commune.Module()
    self.put_json('bro/fam/fam', data=data)
    self.put_json('bro/fam/dawg.json', data=data)
    assert len(self.glob('bro/**')) == 2
    self.rm_json('bro/fam')
    assert len(self.glob('bro/fam/**')) == 0, self.glob('bro/fam/**')


if __name__ == '__main__':
    delete_all_files()
    delete_individual_files()
    delete_directory()
    