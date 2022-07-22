<template>
  <md-card class="md-card-profile">

    <md-card-content>
      <h4 class="card-title">{{$keycloak.fullName}}</h4>
      <h6 class="category text-gray">{{$keycloak.tokenParsed['street']}}, {{$keycloak.tokenParsed['city']}}</h6>
      <h6 class="category text-gray">{{$keycloak.tokenParsed['state']}} {{$keycloak.tokenParsed['zip']}}</h6>
      <md-button v-on:click=$keycloak.logoutFn class="md-round md-success">Log Out</md-button>
    </md-card-content>
  </md-card>
</template>
<script>
export default {
  name: "user-card",
  data() {
    return {};
  }
};
</script>
<style></style>
