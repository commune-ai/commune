import commune as c



def compress_name( name, seperator='.'):
    chunks = name.split(seperator)
    new_chunks = []
    for i, chunk in enumerate(chunks):
        if len(new_chunks)>0:
            if new_chunks[-1] == chunks[i]:
                continue
        new_chunks.append(chunk)
        
    return seperator.join(new_chunks)
                
            
    
print(compress_name('block.block.ray.client.client'))