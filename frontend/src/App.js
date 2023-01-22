import React from 'react';
import Marketplace from './Marketplace';
import Header from './components/header/header';
import './css/dist/output.css'


const items = Array.from({length: 100}, (_, i) => ({item: {title: `Module ${i}`,attributes: ['metrics','bro'], description: 'Description'}}));

const App = () => {
    return (
        <div className="flex flex-col min-h-screen">
            <Header/>
            {/* <Marketplace items={items} /> */}
        </div>
    )
}

export default App;