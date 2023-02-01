
const OS = {
    Mac:0,
    Windows:1,
    Unix:1,
    Linux:1
}

export function user_os(){
    const metadata = navigator.userAgent;
    let os = null;
    if (metadata.search('Windows')!==-1){
         os = OS["Windows"]
    }
    else if (metadata.search('Mac')!==-1){
        os = OS["Mac"]
    }
    else if (metadata.search('X11')!==-1 && !(os.search('Linux')!==-1)){
        os = OS["Unix"]
    }
    else if (metadata.search('Linux')!==-1 && os.search('X11')!==-1){
        os = OS["Linux"]
    }

    return os
    
}