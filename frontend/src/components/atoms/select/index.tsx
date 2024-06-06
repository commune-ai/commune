import { ReactNode } from "react"
import {
  Control,
  Controller,
  FieldError,
  FieldErrorsImpl,
  FieldValues,
  Merge,
  RegisterOptions,
} from "react-hook-form"
import Select, { StylesConfig } from "react-select"

type Options = {
  value: string
  label: string
}
type SelectProps = {
  name: string
  options: Array<Options> | undefined
  placeholder?: string
  errors?: FieldError | Merge<FieldError, FieldErrorsImpl<any>> | undefined
  label?: string
  labelIcon?: ReactNode
  iconComponent?: ReactNode
  isSmall?: boolean
  value?: Options
  isSearchable?: boolean
  isClearable?: boolean
  control: Control<FieldValues, any>
  rules?: RegisterOptions
}

const SelectComp = ({
  name,
  label,
  errors,
  control,
  options,
  placeholder,
  iconComponent,
  labelIcon,
  rules,
  isSmall = false,
  isSearchable = false,
  isClearable = false,
  value: defValue,
}: SelectProps) => {
    console.log('------------', isSmall)
  const errorMessage: any = errors?.message
  const inputClass = errors ? " border-[red]" : "border-[#E8E8E8]"
  const customStyles: StylesConfig = {
    control: (provided) => ({
      ...provided,
      border: "1px solid #D5D8DB",
      height: 50,
      borderRadius: "8px",
      boxShadow: "none !important",
    }),
    option: (provided, state) => ({
      ...provided,
      backgroundColor: state.isSelected ? "#7918F2" : "white",
      color: state.isSelected ? "white" : "black",
    }),
    menu: (provided) => ({
      ...provided,
      marginTop: "2px",
      borderRadius: "18px",
      width: "100%",
      zIndex: 9999,
      boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
    }),
    menuPortal: (base) => ({ ...base, zIndex: 9999 }),
  }

  return (
    <>
      {label && (
        <p className="text-sm flex items-center gap-x-2">
          {labelIcon}
          {label}
        </p>
      )}
      <div
        className={`flex w-full bg-white items-center ${
          !iconComponent ? "ps-0" : "ps-3"
        }  gap-x-4  ${inputClass}`}
      >
        {iconComponent && (
          <div className="h-[30px] w-[30px] flex justify-center items-center text-white bg-primary rounded-sm">
            {iconComponent}
          </div>
        )}
        <Controller
          control={control}
          name={name}
          defaultValue={defValue}
          render={({ field: { onChange, value } }) => (
            <Select
              menuPortalTarget={document.body}
              menuPosition="fixed"
              className="flex-1 rounded-3xl border-none"
              isSearchable={isSearchable}
              isClearable={isClearable}
              options={options}
              value={options?.find(
                (c) => String(c.value) === String(value?.value),
              )}
              onChange={(val: any) => onChange(val)}
              styles={customStyles}
              placeholder={placeholder}
            />
          )}
          rules={rules}
        />
      </div>
      {errorMessage && errorMessage.length > 0 && (
        <small className="text-[11px] text-[#d72c0d] !font-normal">
          {errorMessage}
        </small>
      )}
    </>
  )
}
export default SelectComp
