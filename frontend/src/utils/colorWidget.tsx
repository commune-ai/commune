import React, { useState } from 'react';
import { ColorResult, SketchPicker } from 'react-color';
import { useColorContext } from '@/context/color-widget-provider';

type ColorPickerProps = {
    onChange: (color: string) => void; // Callback function to handle color change
};

const ColorPicker: React.FC<ColorPickerProps> = ({ onChange }) => {
    const [selectedColor, setSelectedColor] = useState<string>('#14313F'); // Initial color
    const { color, changeColor } = useColorContext();

    const handleChange = (color: ColorResult) => {
        setSelectedColor(color.hex);
        onChange(color.hex);
        changeColor(color.hex)
    };

    return (
        <div>
            <SketchPicker color={selectedColor} onChange={handleChange} />
        </div>
    );
};

export default ColorPicker;
