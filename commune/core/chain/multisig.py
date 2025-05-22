

class Multisig:

    def call_multisig(
        self,
        fn: str,
        params: dict[str, Any],
        key: Keypair,
        multisig = None,
        signatories: list[Ss58Address]=None,
        threshold: int = None,
        module: str = "SubspaceModule",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = None,
        sudo: bool = False,
        era: dict[str, int] = None,
    ) -> ExtrinsicReceipt:
        """
        Composes and submits a multisignature call to the network node.

        This method allows the composition and submission of a call that
        requires multiple signatures for execution, known as a multisignature
        call. It supports specifying signatories, a threshold of signatures for
        the call's execution, and an optional era for the call's mortality. The
        call can be a standard extrinsic, a sudo extrinsic for elevated
        permissions, or a multisig extrinsic if multiple signatures are
        required. Optionally, the method can wait for the call's inclusion in a
        block and/or its finalization. Make sure to pass all keys,
        that are part of the multisignature.

        Args:
            fn: The function name to call on the network. params: A dictionary
            of parameters for the call. key: The keypair for signing the
            extrinsic. signatories: List of SS58 addresses of the signatories.
            Include ALL KEYS that are part of the multisig. threshold: The
            minimum number of signatories required to execute the extrinsic.
            module: The module containing the function to call.
            wait_for_inclusion: Whether to wait for the call's inclusion in a
            block. wait_for_finalization: Whether to wait for the transaction's
            finalization. sudo: Execute the call as a sudo (superuser)
            operation. era: Specifies the call's mortality in terms of blocks in
            the format
                {'period': amount_blocks}. If omitted, the extrinsic is
                immortal.

        Returns:
            The receipt of the submitted extrinsic if `wait_for_inclusion` is
            True. Otherwise, returns a string identifier of the extrinsic.

        Raises:
            ChainTransactionError: If the transaction fails.
        """

        # getting the call ready
        with self.get_conn() as substrate:
            # prepares the `GenericCall` object
            call = substrate.call(  # type: ignore
                call_module=module, call_function=fn, call_params=params
            )
            if sudo:
                call = substrate.call(  # type: ignore
                    call_module="Sudo",
                    call_function="sudo",
                    call_params={
                        "call": call.value,  # type: ignore
                    },
                )

            # create the multisig account
            if multisig != None :
                # send the multisig extrinsic
                extrinsic = substrate.create_multisig_extrinsic(  # type: ignore
                    call=call,  # type: ignore
                    keypair=key,
                    multisig_account=multisig,  # type: ignore
                    era=era,  # type: ignore
                )  # type: ignore

            response = substrate.submit_extrinsic(
                extrinsic=extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if wait_for_inclusion:
            if not response.is_success:
                raise ChainTransactionError(
                    response.error_message, response  # type: ignore
                )

        return response


    def get_multisig_path(self, multisig):
        return self.get_path(f'multisig/{multisig}')

    def get_multisig_data(self, multisig):
        if multisig == 'sudo':
            return self.sudo_multisig_data
        if isinstance(multisig, str):
            multisig = c.get(self.get_multisig_path(multisig))
            assert isinstance(multisig, dict)
        return multisig


    def get_multisig(self, multisig):
        if isinstance(multisig, str):
            multisig = self.multisigs().get(multisig)
        if isinstance(multisig, dict):
            return self.multisig(multisig.get('keys'),  multisig.get('threshold'))

        return multisig


    def check_multisig(self, multisig):
        if isinstance(multisig, str):
            multisig = self.get_multisig(multisig)
        if isinstance(multisig, dict):
            keys = multisig.get('signatories', multisig.get('keys'))
            threshold = multisig.get('threshold')
            assert len(keys) >= threshold
            assert len(keys) > 0
            return True
        return False

    def add_multisig(self, name='multisig',  keys=None, threshold=None):
        assert not self.multisig_exists(name)
        if keys == None:
            keys = input('Enter keys (comma separated): ')
            keys = [ k.strip() for k in keys.split(',') ]
        if threshold == None:
            threshold = input('Enter threshold: ')
            threshold = int(threshold)
            assert threshold <= len(keys)

        multisig = {
            'keys': keys,
            'threshold': threshold,
        }
        assert self.check_multisig(multisig)
        path = self.get_multisig_path(name)
        return c.put(path, multisig)

    put_multiisg = add_multisig
    def multisig_exists(self, multisig):
        if isinstance(multisig, str):
            multisig = self.get_multisig(multisig)
        if isinstance(multisig, dict):
            self.check_multisig(multisig)
        return False

    def multisigs(self):
        path = self.get_path(f'multisig')
        paths = c.ls(path)
        multisigs = {}
        for p in paths:
            multisig = c.get(p, None)
            if multisig != None:
                multisigs[p.split('/')[-1].split('.')[-2]] = self.get_multisig_data(multisig)

        # add sudo multisig
        multisigs['sudo'] = self.sudo_multisig_data

        for k, v in multisigs.items():
            if isinstance(v, dict):
                multisig_address = self.multisig(v).ss58_address
                multisigs[k]['address'] = multisig_address
        return multisigs

    mss = multisigs
    