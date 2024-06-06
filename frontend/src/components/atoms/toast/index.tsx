import { toast } from "react-toastify"
import 'react-toastify/dist/ReactToastify.css';

export const infoToast = (message: string) => {
  // toast.info(message, {
  //   position: "top-right",
  //   autoClose: 3000,
  //   progress: "",
  // })

  toast.info(message, {
    position: "top-right",
    autoClose: 5000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "dark",
    // transition: Bounce,
    });
}
export const successToast = (message: string) => {
  // toast.success(message, {
  //   position: "top-right",
  //   autoClose: 3000,
  //   progress: "",
  // })
  toast.success(message, {
    position: "top-right",
    autoClose: 5000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "dark",
    // transition: "Bounce",
    });

}
export const errorToast = (message: string) => {
  // toast.error(message, {
  //   position: "top-right",
  //   autoClose: 3000,
  //   progress: "",
  // })

  toast.error(message, {
    position: "top-right",
    autoClose: 5000,
    hideProgressBar: false,
    closeOnClick: true,
    pauseOnHover: true,
    draggable: true,
    progress: undefined,
    theme: "dark",
    // transition: Bounce,
    });
  
}
