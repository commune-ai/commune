
import React from 'react';
import Link from "next/link"
import { GhostNavbar } from "react-hamburger-menus";
import "react-hamburger-menus/dist/style.css";
import modules from '../../../../modules.json';


const HamburgerModal: React.FC = () => {

  return (

    <GhostNavbar
      styles={
        {
          navigationButton: {
            // borderRadius: "5px",
            //  width: "20px",
            //  height: "20px",  
            backgroundColor: "SteelBlue",
            // opacity: 1,
          },
          navigationBackground: {
            opacity: 0.9,
            backgroundColor: '#06b6d4',
          },
          navigation: { fontFamily: 'Arial, Helvetica, sans-serif' },
        }
      }

      floatButtonY={8}

      floatButtonX={0.1}
    >
      <ul>
        {
          modules.map((item, key) => (
            <li key={key}><Link href={item.url} className='text-black'><p className="text-4xl font-extrabold text-gray-900 dark:text-blue">{item.name}</p></Link></li>
          ))
        }
      </ul>
    </GhostNavbar>
  );
};

export default HamburgerModal;
